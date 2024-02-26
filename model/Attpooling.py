import numpy as np
import torch
import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from transformers import AutoConfig, AutoModel


def get_bert(bert_name):
    if "roberta" in bert_name:
        print("load roberta-base")
        model_config = RobertaConfig.from_pretrained("roberta-base")
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained("roberta-base", config=model_config)

    elif "xlnet" in bert_name:
        print("load xlnet-base-cased")
        model_config = XLNetConfig.from_pretrained("xlnet-base-cased")
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained("xlnet-base-cased", config=model_config)

    else:
        # print('load bert-base-uncased')
        print("load bert-base-chinese")
        model_config = BertConfig.from_pretrained("bert-base-chinese")
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained("bert-base-chinese", config=model_config)
    return bert


class Get_Real_State(nn.Module):
    def __init__(self, bert_name, dropout, attention_dim):
        super(Get_Real_State, self).__init__()
        self.bert = get_bert(bert_name)
        self.attention_dim = attention_dim
        self.pooledout = nn.Sequential(
            nn.Linear(768, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 1)
        )
        self.linear = nn.Linear(768, 176)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0]

        w = self.pooledout(last_hidden_state).float()  # (20,256,1)
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(
            w * last_hidden_state, dim=1
        )  # attention: torch.Size([10, 768])

        attention_embeddings = self.dropout(
            self.linear(attention_embeddings)
        )  # [20,176]

        return attention_embeddings
