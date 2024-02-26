import torch
import os
import sys
import random
from time import time
import math
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class Prediction_State_Herbs(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        num_symptoms,
        num_herbs,
        tcm_l2loss_lambda,
        device,
    ):
        super(Prediction_State_Herbs, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_symptoms = num_symptoms
        # self.herb_weights = herb_weights
        self.num_herbs = num_herbs
        self.tcm_l2loss_lambda = tcm_l2loss_lambda
        self.device = device

        # self.herb_weights = nn.Parameter(torch.Tensor(self.herb_weights))
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        # self.herb_weights = nn.Parameter(torch.Tensor(self.num_herbs, 1))
        self.linear1 = nn.Linear(176, input_size)
        self.linear2 = nn.Linear(hidden_size, 176)
        self.linear3 = nn.Linear(176 * 3, 176)

        self.q = nn.Linear(176 * 2, 176)
        self.k = nn.Linear(176 * 2, 176)
        self.v = nn.Linear(176 * 2, 176)
        self._norm_fact = 1 / math.sqrt(176)

        self.drr_ave = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def loss(
        self,
        nextvisit_state,
        state_afterherbs,
        predict_scores,
        predict_probs,
        ground_t,
        symptoms_embeddings,
        herbs_embeddings,
        all_symptoms_embeddings,
        hh_adj,
        timevisit,
    ):
        # mini_batch= predict_probs.size()[0]
        preherbs_loss = torch.sum(torch.pow((ground_t - predict_probs), 2), 1).mean()

        symptoms_herbs_embed_loss = (
            torch.norm(all_symptoms_embeddings) ** 2 / 2
            + torch.norm(herbs_embeddings) ** 2 / 2
        )
        symptoms_herbs_embed_loss = symptoms_herbs_embed_loss.reshape(1)
        emb_loss = self.tcm_l2loss_lambda * symptoms_herbs_embed_loss

        hh_loss = torch.sigmoid(
            torch.mm(predict_scores.transpose(-1, -2), predict_scores)
        )
        # hh_loss = torch.mm(predict_probs.transpose(-1, -2), predict_probs)
        reg_loss = 0.00001 * hh_loss.unsqueeze(1).mul(hh_adj.to_dense()).sum().reshape(
            1
        )

        if timevisit is not None:
            state_loss = torch.tensor(
                [0.0], dtype=torch.float64, requires_grad=True
            ).to(self.device)
        else:

            state_loss = (
                0.5
                + 0.5
                * (torch.cosine_similarity(nextvisit_state, state_afterherbs, dim=1))
            ).mean()
            state_loss = state_loss.reshape(1)
            # state_loss = state_loss / nextvisit_state.size()[0]

        # batch_loss = state_loss + preherbs_loss + reg_loss
        batch_loss = state_loss + emb_loss + preherbs_loss + reg_loss

        return batch_loss, state_loss, emb_loss, preherbs_loss, reg_loss

    def attention(self, x):
        Q = self.q(x.unsqueeze(1))  # Q: batch_size * seq_len * dim_k
        K = self.k(x.unsqueeze(1))  # K: batch_size * seq_len * dim_k
        V = self.v(x.unsqueeze(1))  # V: batch_size * seq_len * dim_v

        atten = torch.softmax(
            ((torch.bmm(Q, K.transpose(1, 2))) * self._norm_fact), dim=-1
        )  # Q * K.T() # batch_size * seq_len * seq_len
        symptoms_embeddings = torch.bmm(
            atten, V
        )  # Q * K.T() * V # batch_size * seq_len * dim_v
        symptoms_embeddings = symptoms_embeddings.squeeze(1)
        return symptoms_embeddings

    def get_symptomatic_herbal_embed(self, current_state, symptoms, all_embed):
        """
        symptoms: (batch_size, n_symptoms_entities)
        symptom_ids:       the numbers of symptoms for each records
        herb_pos_ids:   (tcm_batch_size)
        herb_neg_ids:   (tcm_batch_size)
        """
        # (n_entities, concat_dim)
        symptoms_ids = torch.where(symptoms == 1)
        symptoms_ids = torch.unique(symptoms_ids[-1])
        symptoms_sets = torch.tensor(range(self.num_symptoms), dtype=torch.long).to(
            self.device
        )

        symptoms_embed = torch.index_select(all_embed, dim=0, index=symptoms_sets)
        # print("symptoms:",symptom_ids)

        symptoms_embed = torch.matmul(symptoms, symptoms_embed)
        symptoms_embeddings = self.attention(
            torch.cat((symptoms_embed, current_state), dim=1)
        )
        # symptoms_embeddings = self.linear1(torch.cat((symptoms_embeddings, current_state), dim=1))
        all_symptoms_embeddings = torch.index_select(
            all_embed, dim=0, index=symptoms_ids
        )
        herbs_sets = torch.tensor(range(self.num_herbs), dtype=torch.long).to(
            self.device
        )
        herbs_embeddings = torch.index_select(all_embed, dim=0, index=herbs_sets)
        return symptoms_embeddings, herbs_embeddings, all_symptoms_embeddings

    def pre_herbs(self, symptoms_embeddings, herbs_embeddings):
        symptoms_embeddings = self.linear2(symptoms_embeddings)
        predict_scores = torch.matmul(
            symptoms_embeddings, herbs_embeddings.transpose(0, 1)
        )
        predict_probs = torch.sigmoid(predict_scores)

        return predict_scores, predict_probs

    def pre_nextvisit_staterep_herbs(
        self, symptoms_embeddings, herbs, herbs_embeddings
    ):

        symptoms_embeddings = self.linear2(symptoms_embeddings)

        herbs_embeddings = torch.matmul(herbs, herbs_embeddings)
        herbs_embeddings = herbs_embeddings.unsqueeze(1)
        # herbs_embeddings = herbs_embeddings.unsqueeze(0).repeat(symptoms_embeddings.size()[0], 1, 1)
        drr_ave = self.drr_ave(herbs_embeddings).squeeze(1)
        state_afterherbs = self.linear3(
            torch.cat((symptoms_embeddings, symptoms_embeddings * drr_ave, drr_ave), 1)
        )

        return state_afterherbs

    def lstmcell(self, x_t, h_t, c_t):
        gates = torch.mm(x_t, self.W) + torch.mm(h_t, self.U) + self.bias  # [2, 512]
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, : self.hidden_size]),  # input
            torch.sigmoid(gates[:, self.hidden_size : self.hidden_size * 2]),  # forget
            torch.tanh(gates[:, self.hidden_size * 2 : self.hidden_size * 3]),
            torch.sigmoid(gates[:, self.hidden_size * 3 :]),  # output
        )

        c_t = f_t * c_t + i_t * g_t  # [2,128]
        h_t = o_t * torch.tanh(c_t)  # [2,128] embed
        return c_t, h_t

    def forward(
        self,
        input_pack,
        symptoms,
        labels,
        all_embed,
        num_visits,
        hh_adj,
        init_states=None,
        train=True,
    ):

        input_seq = input_pack.to(self.device)

        symptoms_seq = symptoms.to(self.device)
        labels_herbs = labels.to(self.device)
        minibatch_size = num_visits.to(self.device)
        if init_states is None:
            h_t, c_t = (
                torch.zeros(minibatch_size[0], self.hidden_size).to(
                    self.device
                ),  # [2,128]
                torch.zeros(minibatch_size[0], self.hidden_size).to(self.device),
            )  # [2,128]
        else:
            h_t, c_t = init_states

        hidden_seq = []

        start = 0
        end = minibatch_size[0]
        realistic_state = input_seq.clone().to(self.device)
        Predict_herbs_scores = []
        Predict_state_scores = []
        (
            time_tcm_loss,
            time_state_loss,
            time_emb_loss,
            time_preherbs_loss,
            time_reg_loss,
        ) = (0, 0, 0, 0, 0)
        time_tcm_len, time_state_len, time_emb_len, time_preherbs_len, time_reg_len = (
            [],
            [],
            [],
            [],
            [],
        )
        if train:
            for t in range(len(minibatch_size)):
                current_state = realistic_state[: minibatch_size[t], :]
                # real_state = input_seq[:b, :]
                s_t = symptoms_seq[start:end, :]
                herbs_t = labels_herbs[start:end, :]

                symptoms_embeddings, herbs_embeddings, all_symptoms_embeddings = (
                    self.get_symptomatic_herbal_embed(current_state, s_t, all_embed)
                )
                c_t, h_t = self.lstmcell(symptoms_embeddings, h_t, c_t)

                predict_scores, predict_probs = self.pre_herbs(h_t, herbs_embeddings)
                state_afterherbs = self.pre_nextvisit_staterep_herbs(
                    h_t, herbs_t, herbs_embeddings
                )

                if t < len(minibatch_size) - 1:
                    start += minibatch_size[t]
                    end += minibatch_size[t + 1]
                    nextvisit_real_state = input_seq[start:end, :]
                    c_t = c_t[: minibatch_size[t + 1], :]
                    h_t = h_t[: minibatch_size[t + 1], :]
                    state_afterherbs = state_afterherbs[: minibatch_size[t + 1], :]
                    realistic_state = state_afterherbs

                    (
                        batch_loss,
                        batch_state_loss,
                        batch_emb_loss,
                        batch_preherbs_loss,
                        batch_reg_loss,
                    ) = self.loss(
                        nextvisit_real_state,
                        state_afterherbs,
                        predict_scores,
                        predict_probs,
                        herbs_t,
                        symptoms_embeddings,
                        herbs_embeddings,
                        all_symptoms_embeddings,
                        hh_adj,
                        timevisit=None,
                    )
                else:
                    nextvisit_real_state = input_seq[start:end, :]
                    (
                        batch_loss,
                        batch_state_loss,
                        batch_emb_loss,
                        batch_preherbs_loss,
                        batch_reg_loss,
                    ) = self.loss(
                        nextvisit_real_state,
                        state_afterherbs,
                        predict_scores,
                        predict_probs,
                        herbs_t,
                        symptoms_embeddings,
                        herbs_embeddings,
                        all_symptoms_embeddings,
                        hh_adj,
                        timevisit="last",
                    )

                time_tcm_loss += batch_loss
                time_state_loss += batch_state_loss
                time_emb_loss += batch_emb_loss
                time_preherbs_loss += batch_preherbs_loss
                time_reg_loss += batch_reg_loss

                time_tcm_len.append(batch_loss)
                time_state_len.append(batch_state_loss)
                time_emb_len.append(batch_emb_loss)
                time_preherbs_len.append(batch_preherbs_loss)
                time_reg_len.append(batch_reg_loss)

            time_tcm_loss /= len(time_tcm_len)
            time_state_loss /= len(time_state_len)
            time_emb_loss /= len(time_emb_len)
            time_preherbs_loss /= len(time_preherbs_len)
            time_reg_loss /= len(time_reg_len)

            return (
                time_tcm_loss,
                time_state_loss,
                time_emb_loss,
                time_preherbs_loss,
                time_reg_loss,
            )

        else:
            for t in range(len(minibatch_size)):
                current_state = realistic_state[: minibatch_size[t], :]
                # real_state = input_seq[:b, :]
                s_t = symptoms_seq[start:end, :]
                # herbs_t = labels_herbs[start:end, :]

                symptoms_embeddings, herbs_embeddings, all_symptoms_embeddings = (
                    self.get_symptomatic_herbal_embed(current_state, s_t, all_embed)
                )
                c_t, h_t = self.lstmcell(symptoms_embeddings, h_t, c_t)

                predict_scores, predict_probs = self.pre_herbs(h_t, herbs_embeddings)
                state_afterherbs = self.pre_nextvisit_staterep_herbs(
                    h_t, predict_probs, herbs_embeddings
                )
                Predict_herbs_scores.append(predict_probs)

                if t < len(minibatch_size) - 1:
                    start += minibatch_size[t]
                    end += minibatch_size[t + 1]
                    nextvisit_real_state = input_seq[start:end, :]
                    c_t = c_t[: minibatch_size[t + 1], :]
                    h_t = h_t[: minibatch_size[t + 1], :]
                    state_afterherbs = state_afterherbs[: minibatch_size[t + 1], :]
                    realistic_state = state_afterherbs

                    prediction_state = torch.cosine_similarity(
                        nextvisit_real_state, state_afterherbs, dim=1
                    )
                    Predict_state_scores.append(prediction_state)

            return Predict_herbs_scores, Predict_state_scores
