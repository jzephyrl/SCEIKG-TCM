import torch
import os
import sys
import random
from time import time
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from model.aggregator import Aggregator
from model.attpooling import Get_Real_State
from model.predictionHerbs import Prediction_State_Herbs
from torch.nn.utils.rnn import (
    pack_sequence,
    pad_sequence,
    pad_packed_sequence,
    pack_padded_sequence,
)
from audtorch.metrics.functional import pearsonr


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.0)


class Consecutive_visit_Model(nn.Module):

    def __init__(
        self,
        args,
        n_entities,
        n_relations,
        num_symptoms,
        num_herbs,
        device,
        A_in=None,
        hh_adj=None,
        symptoms_pre_embed=None,
        herbs_pre_embed=None,
    ):

        super(Consecutive_visit_Model, self).__init__()
        # self.attention_dim = attentiton_dim
        self.bert_name = args.bert_name
        self.dropout = args.dropout
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.num_symptoms = num_symptoms
        self.num_herbs = num_herbs
        self.device = device

        self.use_pretrain = args.use_pretrain

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.tcm_l2loss_lambda = args.tcm_l2loss_lambda

        self.entity_symptom_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.epsilon = 2.0

        self.trans_M = nn.Parameter(
            torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim)
        )

        self.gamma = nn.Parameter(torch.Tensor([12.0]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.embed_dim]),
            requires_grad=False,
        )

        if (
            (self.use_pretrain == 1)
            and (symptoms_pre_embed is not None)
            and (herbs_pre_embed is not None)
        ):
            other_entity_embed = nn.Parameter(
                torch.Tensor(
                    self.n_entities
                    - herbs_pre_embed.shape[0]
                    - symptoms_pre_embed.shape[0],
                    self.embed_dim,
                )
            )
            nn.init.uniform_(other_entity_embed)
            entity_symptom_embed = torch.cat(
                [herbs_pre_embed, symptoms_pre_embed, other_entity_embed], dim=0
            )
            self.entity_symptom_embed.weight = nn.Parameter(entity_symptom_embed)
        else:
            nn.init.uniform_(
                self.entity_symptom_embed.weight,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item(),
            )

        nn.init.uniform_(
            self.relation_embed.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(
                    self.conv_dim_list[k],
                    self.conv_dim_list[k + 1],
                    self.mess_dropout[k],
                    self.aggregation_type,
                )
            )

        self.A_in = nn.Parameter(
            torch.sparse.FloatTensor(self.n_entities, self.n_entities)
        )
        self.hh_adj = nn.Parameter(
            torch.sparse.FloatTensor(self.num_herbs, self.num_herbs)
        )
        if A_in is not None:
            self.A_in.data = A_in
            self.hh_adj.data = hh_adj

        self.A_in.requires_grad = False
        self.hh_adj.requires_grad = False


        self.get_real_state = Get_Real_State(
            self.bert_name, self.dropout, self.input_size
        )

        self.pre_state_herbs = Prediction_State_Herbs(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            self.num_symptoms,
            self.num_herbs,
            self.tcm_l2loss_lambda,
            self.device,
        )

    def calc_tcm_embeddings(self):
        """ """
        # print("self.entity_symptom_embed1:",self.entity_symptom_embed.weight[0])
        ego_embed = self.entity_symptom_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)  # (n_entities, concat_dim)
        return all_embed

    def calc_tcm_loss(
        self, input_ids, attention_mask, token_type_ids, symptoms, labels, num_visits
    ):

        all_embed = self.calc_tcm_embeddings()
        # print("all_embed1:",all_embed[0][:5])

        num_visits = num_visits.detach().cpu().numpy()
        input_pack = pack_padded_sequence(
            input_ids, lengths=num_visits, batch_first=True, enforce_sorted=False
        )
        # print("input_pack:",input_pack[1])
        attention_pack = pack_padded_sequence(
            attention_mask, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]
        token_type_pack = pack_padded_sequence(
            token_type_ids, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]
        symptoms = pack_padded_sequence(
            symptoms, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]
        # print("symptoms:",symptoms)
        # print("1:",torch.where(symptoms==1))
        labels = pack_padded_sequence(
            labels, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]

        everytime_state = self.get_real_state(
            input_pack[0], attention_mask=attention_pack, token_type_ids=token_type_pack
        )  # [20,768]
        # num_visits = torch.from_numpy(num_visits)
        # print(num_visits)
        num_visits = input_pack[1]
        # print("2:",num_visits)

        (
            time_tcm_loss,
            time_state_loss,
            time_emb_loss,
            time_preherbs_loss,
            time_reg_loss,
        ) = self.pre_state_herbs(
            everytime_state,
            symptoms,
            labels,
            all_embed,
            num_visits,
            self.hh_adj,
            init_states=None,
            train=True,
        )

        # print("all_embed2:",all_embed[0][:5])

        return (
            time_tcm_loss,
            time_state_loss,
            time_emb_loss,
            time_preherbs_loss,
            time_reg_loss,
        )

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """

        r_embed = self.relation_embed(r)
        W_r = self.trans_M[r]
        h_embed = self.entity_symptom_embed(h)
        pos_t_embed = self.entity_symptom_embed(pos_t)
        neg_t_embed = self.entity_symptom_embed(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        # (kg_batch_size, relation_dim)

        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)
        # (kg_batch_size, relation_dim)

        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]
        r_mul_h = r_mul_h / (self.embedding_range.item() / pi)
        r_embed = r_embed / (self.embedding_range.item() / pi)
        r_mul_pos_t = r_mul_pos_t / (self.embedding_range.item() / pi)
        r_mul_neg_t = r_mul_neg_t / (self.embedding_range.item() / pi)

        pos_score = torch.abs(torch.sin(r_mul_h + r_embed - r_mul_pos_t))
        pos_score = self.gamma.item() - pos_score.sum(dim=1) * self.modulus

        neg_score = torch.abs(torch.sin(r_mul_h + r_embed - r_mul_neg_t))
        neg_score = self.gamma.item() - neg_score.sum(dim=1) * self.modulus

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = (
            _L2_loss_mean(r_mul_h)
            + _L2_loss_mean(r_embed)
            + _L2_loss_mean(r_mul_pos_t)
            + _L2_loss_mean(r_mul_neg_t)
        )
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss


        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_symptom_embed.weight[h_list]
        t_embed = self.entity_symptom_embed.weight[t_list]

        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)

        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(
                batch_h_list, batch_t_list, r_idx
            )
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        hh_rows = []
        hh_cols = []
        hh_values = []

        for i, j, v in zip(
            rows.detach().cpu().numpy(),
            cols.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
        ):
            if i < self.num_herbs and j < self.num_herbs:
                hh_rows.append(i)
                hh_cols.append(j)
                hh_values.append(v)

        hh_values = torch.FloatTensor(hh_values)
        hh_rows = torch.LongTensor(hh_rows)
        hh_cols = torch.LongTensor(hh_cols)
        hh_indices = torch.stack([hh_rows, hh_cols])
        # hh_i = torch.LongTensor(hh_indices)
        hh_adj = torch.sparse.FloatTensor(
            hh_indices, hh_values, torch.Size((self.num_herbs, self.num_herbs))
        )
        hh_adj = torch.sparse.softmax(hh_adj.cpu(), dim=1)

        # print("---------------------")
        indices = torch.stack([rows, cols])
    
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)


        self.hh_adj.data = hh_adj.to(device)
        self.A_in.data = A_in.to(device)


    def calc_score(
        self, input_ids, attention_mask, token_type_ids, symptoms, labels, num_visits
    ):
        """
        symptom_ids:  (n_symptoms)
        herb_ids:  (n_herbs)
        """

        all_embed = self.calc_tcm_embeddings()
        num_visits = num_visits.detach().cpu().numpy()
        input_pack = pack_padded_sequence(
            input_ids, lengths=num_visits, batch_first=True, enforce_sorted=False
        )
        attention_pack = pack_padded_sequence(
            attention_mask, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]
        token_type_pack = pack_padded_sequence(
            token_type_ids, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]
        symptoms = pack_padded_sequence(
            symptoms, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]
        labels = pack_padded_sequence(
            labels, lengths=num_visits, batch_first=True, enforce_sorted=False
        )[0]

        everytime_state = self.get_real_state(
            input_pack[0], attention_mask=attention_pack, token_type_ids=token_type_pack
        )  # [20,768]

        # num_visits = torch.from_numpy(num_visits)
        num_visits = input_pack[1]
        batch_herbs_scores, batch_state_scores = self.pre_state_herbs(
            everytime_state,
            symptoms,
            labels,
            all_embed,
            num_visits,
            self.hh_adj,
            init_states=None,
            train=False,
        )
        print("self.hh_adj:", self.hh_adj)
        print("self.A_in.data:", self.A_in.data)

        return (
            batch_herbs_scores,
            batch_state_scores,
            labels,
            self.hh_adj,
            self.A_in.data,
        )

    def forward(self, *input, mode):
        if mode == "train_tcm":
            return self.calc_tcm_loss(*input)
        if mode == "train_kg":
            return self.calc_kg_loss(*input)
        if mode == "update_att":
            return self.update_attention(*input)
        if mode == "evaluate":
            return self.calc_score(*input)
