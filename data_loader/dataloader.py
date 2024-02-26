import os
import random
import collections
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import pdb
from torch.utils.data import Dataset
from torch.nn.utils.rnn import (
    pack_sequence,
    pad_sequence,
    pad_packed_sequence,
    pack_padded_sequence,
)

from data_loader.dataloader_base import DataLoaderBase


class DataLoaderConsModel(DataLoaderBase):
    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.tcm_batch_size = args.tcm_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        self.construct_data()
        self.print_info(logging)

        self.convert_adjcoo2tensor()

    def construct_data(self):
        """
        constrcut tcm datasets and add the tcm triples into the kg
        constrcut kg data to train
        self.trainTimeDataset used to train the model of recommend
        self.train_kg_dict used to train the triples
        self.train_relation_dict used to constrcut the adj

        """

        # Construct the dataset used to recommend herbs
        self.train_records = self.train_records
        self.train_symptoms = self.train_symptoms
        self.train_labels = self.train_labels

        self.test_records = self.test_records
        self.test_symptoms = self.test_symptoms
        self.test_labels = self.test_labels
        # self.herb_weights = self.herb_weights


        # concat
        self.kg_train_data = self.train_kg_data
        self.adj_norm = self.adj_norm
        self.num_relations = max(self.kg_train_data["r"]) + 1
        self.num_entities = (
            max(max(self.kg_train_data["h"]), max(self.kg_train_data["t"])) + 1
        )
        self.num_tcm2kg_entities = self.num_entities
        self.num_kg_train = len(self.kg_train_data)

        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)


        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_adjcoo2tensor(self):
        hh_rows = []
        hh_cols = []
        hh_values = []

        for i, j, v in zip(self.adj_norm.row, self.adj_norm.col, self.adj_norm.data):
            if i < self.num_herbs and j < self.num_herbs:
                hh_rows.append(i)
                hh_cols.append(j)
                hh_values.append(v)

        hh_indices = np.vstack((hh_rows, hh_cols))
        hh_i = torch.LongTensor(hh_indices)
        hh_v = torch.FloatTensor(hh_values)
        self.hh_adj = torch.sparse.FloatTensor(
            hh_i, hh_v, torch.Size((self.num_herbs, self.num_herbs))
        )

        values = self.adj_norm.data
        indices = np.vstack((self.adj_norm.row, self.adj_norm.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = self.adj_norm.shape
        self.A_in = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # print("self.A_in:", self.A_in)

    def print_info(self, logging):
        logging.info("num_symptoms:           %d" % self.num_symptoms)
        logging.info("num_herbs:           %d" % self.num_herbs)
        logging.info("num_entities:        %d" % self.num_entities)
        logging.info("num_tcm2kg_entities:  %d" % self.num_tcm2kg_entities)
        logging.info("num_relations:       %d" % self.num_relations)
        logging.info("num_h_list:          %d" % len(self.h_list))
        logging.info("num_t_list:          %d" % len(self.t_list))
        logging.info("num_r_list:          %d" % len(self.r_list))
        logging.info("num_tcm_train:        %d" % self.num_tcm_train)
        logging.info("num_tcm_test:         %d" % self.num_tcm_test)
        logging.info("num_kg_train:        %d" % self.num_kg_train)


class MDataset(Dataset):
    def __init__(
        self, Records, Symptoms, Herbs, num_symptoms, num_herbs, tokenizer, max_length
    ):
        self.records = Records
        self.max_length = max_length
        self.Symptoms = Symptoms
        self.Herbs = Herbs
        self.num_symptoms = num_symptoms
        self.num_herbs = num_herbs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        if len(self.records[index]) == 1:
            encode_info = self.tokenizer.encode_plus(
                (
                    "filling empty"
                    if len(self.records[index][0]) == 0
                    else self.records[index][0]
                ),
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

            patient_ids = [encode_info["input_ids"]]
            patient_type = [encode_info["token_type_ids"]]
            patient_att = [encode_info["attention_mask"]]

            # one-hot
            onehot_symptoms = np.zeros((1, self.num_symptoms), dtype=float)
            onehot_herbs = np.zeros((1, self.num_herbs), dtype=float)
            syms = self.Symptoms[index][0]
            herbs = self.Herbs[index][0]

            for uid in syms:
                onehot_symptoms[0][int(uid)] = 1
                # symptoms_set.add(int(uid))
            for herb in herbs:
                onehot_herbs[0][int(herb)] = 1
                # herbs_set.add(int(herb))

        else:
            patient_ids = []
            patient_type = []
            patient_att = []
            for j in range(len(self.records[index])):
                encode_info = self.tokenizer.encode_plus(
                    (
                        "filling empty"
                        if len(self.records[index][j]) == 0
                        else self.records[index][j]
                    ),
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                patient_ids.append(encode_info["input_ids"])
                patient_type.append(encode_info["token_type_ids"])
                patient_att.append(encode_info["attention_mask"])


            onehot_symptoms = np.zeros(
                (len(self.Symptoms[index]), self.num_symptoms), dtype=float
            )
            onehot_herbs = np.zeros(
                (len(self.Symptoms[index]), self.num_herbs), dtype=float
            )
            for id in range(len(self.Symptoms[index])):
                syms = self.Symptoms[index][id]
                herbs = self.Herbs[index][id]
                # pos_herb = self.sample_pos_herbs_for_u(herbs,1)
                # neg_herb = self.sample_neg_herbs_for_u(herbs,1)
                for uid in syms:
                    onehot_symptoms[id][int(uid)] = 1
                    # symptoms_set.add(int(uid))
                for herb in herbs:
                    onehot_herbs[id][int(herb)] = 1
                    # herbs_set.add(int(herb))

        return (
            torch.LongTensor(patient_ids),
            torch.LongTensor(patient_type),
            torch.LongTensor(patient_att),
            torch.Tensor(onehot_symptoms),
            torch.Tensor(onehot_herbs),
        )

    @staticmethod
    def collate_fn(train_data):

        train_data.sort(key=lambda train_data: len(train_data[0]), reverse=True)
        # print("train_data:",train_data)
        data_length = [len(data[0]) for data in train_data]
        # print("length:",data_length)
        input = [data[0] for data in train_data]
        token = [data[1] for data in train_data]
        mask = [data[2] for data in train_data]
        symptoms = [data[3] for data in train_data]
        labels = [data[4] for data in train_data]


        input = pad_sequence(input, batch_first=True, padding_value=0)
        # print("input:",input)
        token = pad_sequence(token, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        symptoms = pad_sequence(symptoms, batch_first=True, padding_value=0)
        onehot_labels = pad_sequence(labels, batch_first=True, padding_value=0)
        # labels = pad_sequence(herbs, batch_first=True, padding_value = 0)
        data_length = torch.tensor(data_length)
        # print(input.size())
        return input, token, mask, symptoms, onehot_labels, data_length
