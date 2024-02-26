import os
import time
import random
import collections
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle


class DataLoaderBase(object):
    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.max_length = args.max_length
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.data_dir = os.path.join(args.data_dir, args.data_name)

        # construct train and test dataset, timeseries
        self.train_symptoms_file = os.path.join(self.data_dir, "train_symptoms.txt")
        self.test_symptoms_file = os.path.join(self.data_dir, "test_symptoms.txt")

        self.trian_records_file = os.path.join(self.data_dir, "train_records.txt")
        self.test_records_file = os.path.join(self.data_dir, "test_records.txt")

        self.train_labels_file = os.path.join(self.data_dir, "train_label.txt")
        self.test_labels_file = os.path.join(self.data_dir, "test_label.txt")

        # read data
        (
            self.train_records,
            self.train_symptoms,
            self.train_labels,
            self.train_num_symptoms,
            self.train_num_herbs,
        ) = self.load_data(
            self.trian_records_file, self.train_symptoms_file, self.train_labels_file
        )

        (
            self.test_records,
            self.test_symptoms,
            self.test_labels,
            self.test_num_symptoms,
            self.test_num_herbs,
        ) = self.load_data(
            self.test_records_file, self.test_symptoms_file, self.test_labels_file
        )

        self.statistic_tcm()

        self.train_kg_data_file = os.path.join(self.data_dir, "train_kg_data.txt")
        self.norm_adj_file = os.path.join(self.data_dir, "norm_adj.pkl")

        self.train_kg_data, self.adj_norm = self.load_kg_data(
            self.train_kg_data_file, self.norm_adj_file
        )
        # self.train_file = os.path.join(self.data_dir,'train_sym_herbs.txt')
        # self.test_file = os.path.join(self.data_dir, 'test_sym_herbs.txt')
        # self.kg_file = os.path.join(self.data_dir, "KG_inverse_map.txt")
        # self.kg_file = os.path.join('../datasets/tcm/', "KG_map1.txt")
        # construct the relation of symptom and herbs on data, which not time series data
        # self.train_triples, self.herb_weights = self.dok_matrix(self.train_file, self.test_file)

        if self.use_pretrain == 1:
            self.load_pretrained_data()


    def load_data(self, filename1, filename2, filename3):
        """
        filename1: text description of the patient
        filename2: the nodes of symptoms of the patient
        filename3: the herbs of the patient
        len(Records)=len(Symptoms)=len(Herbs): represent how many patients
        len(Records[i])=len(Symptoms[i])=len(Herbs[i]): represent the number of consecutive visits to a patient
        num_symptoms: represent the number of symptom nodes to train datasets or test datasets (note: start from zero...)
        num_herbs: represent the number of herb nodes to train datasets or test dataset
        """
        # read text
        with open(filename1, "r") as fin:
            Records = []
            for line in fin.readlines():
                line = line.strip().split("\t")
                text = []
                # 一个病人看病的次数，每次看病的病历描述最大词为256
                for i in range(1, len(line)):
                    review = " ".join(line[i].split()[: self.max_length])
                    text.append(review)
                Records.append(text)

        # 读取症状节点的数据,要把Symptoms改成one-hot形式
        symptoms_set = set()
        with open(filename2, "r") as fin:
            Symptoms = []
            for line in fin.readlines():
                line = line.strip().split("\t")
                symptoms = []
                for i in range(1, len(line)):
                    tmpS = line[i].split(",")
                    try:
                        symptoms_ids = [int(i) for i in tmpS]
                        for symptoms_id in symptoms_ids:
                            symptoms_set.add(symptoms_id)
                    except Exception:
                        continue
                    symptoms.append(symptoms_ids)
                Symptoms.append(symptoms)

        # 读取医生开的草药即labels
        herbs_set = set()
        with open(filename3, "r") as fin:
            Herbs = []
            for line in fin.readlines():
                line = line.strip().split("\t")
                herbs = []
                for i in range(1, len(line)):
                    tmpH = line[i].split(",")
                    try:
                        herbs_ids = [int(i) for i in tmpH]
                        for herbs_id in herbs_ids:
                            herbs_set.add(herbs_id)
                    except Exception:
                        continue
                    herbs.append(herbs_ids)
                Herbs.append(herbs)

        assert len(Records) == len(Symptoms) == len(Herbs)
        num_symptoms = max(symptoms_set)  # 6567
        num_herbs = max(herbs_set)  # 385

        return Records, Symptoms, Herbs, num_symptoms, num_herbs

    def statistic_tcm(self):
        """
        self.num_symptoms: represent the number of symptom nodes to all patients (note: start from zero...)
        self.num_herbs: represent the number of herb nodes to all patients
        self.num_tcm_train: represent how many train datasets (patients of train datasets)
        self.num_tcm_test: represent how many test datasets (patients of test datasets)
        """
        self.num_symptoms = (
            max(self.train_num_symptoms, self.test_num_symptoms) + 1
        )  
        self.num_herbs = max(self.train_num_herbs, self.test_num_herbs) + 1  # 387
        self.num_tcm_train = len(self.train_records)
        self.num_tcm_test = len(self.test_records)

    def load_kg_data(self, filename1, filename2):
        kg_train_data = pd.read_csv(
            filename1, sep="\t", names=["h", "r", "t"], engine="python"
        )
       
        kg_train_data = kg_train_data.drop_duplicates()
        with open(filename2, "rb") as f:
            norm_adj = pickle.load(f)
        return kg_train_data, norm_adj

    def sample_pos_triples_for_h(self, kg_dict, head, num_sample_pos_triples):
        pos_triples = kg_dict[head]
        num_pos_triples = len(pos_triples)
        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == num_sample_pos_triples:
                break
            pos_triple_idx = np.random.randint(low=0, high=num_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]
            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(
        self, kg_dict, head, relation, num_sample_neg_triples, highest_neg_idx
    ):
        pos_triples = kg_dict[head]
        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == num_sample_neg_triples:
                break
            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        # print("len:",len(exist_heads))
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail
            neg_tail = self.sample_neg_triples_for_h(
                kg_dict, h, relation[0], 1, highest_neg_idx
            )
            batch_neg_tail += neg_tail
        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def load_pretrained_data(self):
        pre_model = "lstm_gcn"
        pretrain_path = "%s/%s/%s.npz" % (
            self.pretrain_embedding_dir,
            self.data_name,
            pre_model,
        )
        pretrain_data = np.load(pretrain_path)
        self.symptoms_pre_embed = pretrain_data["symptoms_embed"]
        self.herbs_pre_embed = pretrain_data["herbs_embed"]
        # print("self.symptoms_pre_embed.shape[0]:",self.symptoms_pre_embed.shape)
        # print("self.symptoms_pre_embed.shape[0]:",self.herb_pre_embed.shape)
        # assert self.symptoms_pre_embed.shape[0] == self.num_symptoms
        assert self.herbs_pre_embed.shape[0] == self.num_herbs
        assert self.symptoms_pre_embed.shape[1] == self.args.embed_dim
        assert self.herbs_pre_embed.shape[1] == self.args.embed_dim
