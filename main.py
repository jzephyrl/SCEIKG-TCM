import numpy as np
import torch
import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertConfig, BertModel
from torch.utils.data import DataLoader

from utils.metrics import *
from data_loader.dataloader import DataLoaderConsModel, MDataset
from parser.parse_tcm import *
from utils.log_helper import *
from utils.model_helper import *
from model.ConsecutiveModel import Consecutive_visit_Model

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# import ipdb
from scipy.sparse import coo_matrix
import os


# load data
def load_data(args):
    log_save_id, save_dir = create_log_id(args.save_dir)
    logging_config(
        folder=save_dir, name="log{:d}".format(log_save_id), no_console=False
    )
    logging.info(args)

    # load data
    data = DataLoaderConsModel(args, logging)

    # construct train dataset and test dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
    train_data = MDataset(
        data.train_records,
        data.train_symptoms,
        data.train_labels,
        data.num_symptoms,
        data.num_herbs,
        tokenizer,
        args.max_length,
    )
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=args.rank)
    test_data = MDataset(
        data.test_records,
        data.test_symptoms,
        data.test_labels,
        data.num_symptoms,
        data.num_herbs,
        tokenizer,
        args.max_length,
    )

    trainloader = DataLoader(
        train_data,
        batch_size=data.tcm_batch_size,
        collate_fn=MDataset.collate_fn,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )
    testloader = DataLoader(
        test_data,
        batch_size=data.test_batch_size,
        collate_fn=MDataset.collate_fn,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    return data, trainloader, testloader, save_dir


def evaluate(model, dataloader, testloader, Ks, device, save_dir, save):

    # test_batch_size = dataloader.test_batch_size
    # testTimeDataset = dataloader.testTimeDataset
    num_tcm2kg_entities = dataloader.num_tcm2kg_entities
    num_herbs = dataloader.num_herbs
    herb_ids = torch.arange(num_herbs, dtype=torch.long).to(device)
    tcm_scores = []
    metric_names = ["precision", "recall", "ndcg", "f1"]
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with torch.no_grad():
        model.eval()
        herbs_scores = []
        state_scores = []
        groundtruth_onehot_herbs = []
        # groundtruth_herbs = []
        for step, input in enumerate(testloader):
            input_ids = input[0].to(device)
            token_type_ids = input[1].to(device)
            attention_mask = input[2].to(device)
            test_symptoms = input[3].to(device)
            groundth_onehot_labels = input[4].to(device)
            num_visits = input[5].to(device)
            labels = herb_ids.to(device)
            # groundth_labels = data[5]

            batch_herbs_scores, batch_state_scores, gt_onehot_labels, hh_adj, A_in = (
                model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    test_symptoms,
                    groundth_onehot_labels,
                    num_visits,
                    mode="evaluate",
                )
            )

            herbs_scores.append(
                torch.cat([item.cpu().detach() for item in batch_herbs_scores], 0)
            )
            # state_scores.append(torch.cat([item.cpu().detach() for item in batch_state_scores], 0))
            groundtruth_onehot_herbs.append(gt_onehot_labels.cpu().numpy())

        herbs_scores = torch.cat(herbs_scores, 0)
        # state_scores = torch.cat(state_scores, 0)
        groundtruth_onehot_herbs = np.concatenate(groundtruth_onehot_herbs, axis=0)
        batch_metrics, predict_rank_indices = calc_metrics_at_k(
            herbs_scores, groundtruth_onehot_herbs, Ks
        )

        dense_tensor = hh_adj.to_dense()
        numpy_array = dense_tensor.cpu().detach().numpy()

        ground_herbs = []
        predict_herbs = []
        ground_rank_indices = [
            (np.where(groundtruth_onehot_herbs[i] == 1)[0])
            for i in range(len(groundtruth_onehot_herbs))
        ]
        ground_herbs.extend(ground_rank_indices)
        predict_herbs.extend(predict_rank_indices.tolist())

        # tcm_scores.append(batch_herbs_scores.numpy())
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m].append(batch_metrics[k][m])
        # pbar.update(1)

    if save != None:
        save_label(args, ground_herbs, predict_herbs, save_dir)

    # tcm_scores = np.concatenate(tcm_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return herbs_scores, metrics_dict


def train(args, data, trainloader, testloader, save_dir):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # GPU / CPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    if args.use_pretrain == 1:
        symptoms_pre_embed = torch.tensor(data.symptoms_pre_embed)
        herbs_pre_embed = torch.tensor(data.herbs_pre_embed)
    else:
        symptoms_pre_embed, herbs_pre_embed = None, None

    # construct the model consecutive visits
    model = Consecutive_visit_Model(
        args,
        data.num_entities,
        data.num_relations,
        data.num_symptoms,
        data.num_herbs,
        device,
        data.A_in,
        data.hh_adj,
        symptoms_pre_embed,
        herbs_pre_embed,
    ).to(device)

    logging.info(model)

    if args.use_pretrain == 2:
        pretrain_model_path = save_dir
        model = load_model(model, pretrain_model_path)

    tcm_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics

    best_epoch, best_recall = -1, 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_mid = Ks[1]
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {
        k: {"precision": [], "recall": [], "ndcg": [], "f1": []} for k in Ks
    }

    # Training
    for epoch in tqdm(range(1, args.n_epoch + 1)):
        print(f"start to train tcm...")
        time0 = time()
        model.train()
        # train tcm
        time1 = time()
        tcm_total_loss, tcm_state_loss, tcm_herbs_loss, tcm_emb_loss, tcm_reg_loss = (
            0,
            0,
            0,
            0,
            0,
        )
        # num_tcm_batch = data.num_tcm_train // data.tcm_batch_size + 1

        for step, input in tqdm(enumerate(trainloader)):
            # gc.collect()
            # torch.cuda.empty_cache()
            time2 = time()

            # max_length = data[0].size()[-1]
            input_ids = input[0].to(device)
            token_type_ids = input[1].to(device)
            attention_mask = input[2].to(device)
            symptoms = input[3].to(device)
            labels = input[4].to(device)
            num_visits = input[5].to(device)

            (
                time_tcm_loss,
                time_state_loss,
                time_emb_loss,
                time_preherbs_loss,
                time_reg_loss,
            ) = model(
                input_ids,
                attention_mask,
                token_type_ids,
                symptoms,
                labels,
                num_visits,
                mode="train_tcm",
            )

            if np.isnan(time_tcm_loss.cpu().detach().numpy()):
                logging.info(
                    "ERROR (tcm Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.".format(
                        epoch, (step + 1), len(trainloader)
                    )
                )
                sys.exit()

            tcm_optimizer.zero_grad()
            time_tcm_loss.backward()
            tcm_optimizer.step()
            tcm_total_loss += time_tcm_loss.item()
            tcm_state_loss += time_state_loss.item()
            tcm_herbs_loss += time_preherbs_loss.item()
            tcm_emb_loss += time_emb_loss.item()
            tcm_reg_loss += time_reg_loss.item()

            if (step % args.tcm_print_every) == 0:
                logging.info(
                    "tcm Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f} | Iter Herbs Loss {:.4f} | Iter Herbs Mean Loss {:.4f} | Iter State Mean Loss {:.4f}".format(
                        epoch,
                        (step + 1),
                        len(trainloader),
                        time() - time2,
                        time_tcm_loss.item(),
                        tcm_total_loss / (step + 1),
                        time_preherbs_loss.item(),
                        tcm_herbs_loss / (step + 1),
                        tcm_state_loss / (step + 1),
                    )
                )
        logging.info(
            "tcm Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f} | Iter Herbs Mean Loss {:.4f} | Iter State Mean Loss {:.4f}".format(
                epoch,
                len(trainloader),
                time() - time1,
                tcm_total_loss / len(trainloader),
                tcm_herbs_loss / len(trainloader),
                tcm_state_loss / len(trainloader),
            )
        )

        time3 = time()
        kg_total_loss = 0
        num_kg_batch = data.num_kg_train // data.kg_batch_size + 1

        for step in range(1, num_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = (
                data.generate_kg_batch(
                    data.train_kg_dict, data.kg_batch_size, data.num_tcm2kg_entities
                )
            )
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(
                kg_batch_head,
                kg_batch_relation,
                kg_batch_pos_tail,
                kg_batch_neg_tail,
                mode="train_kg",
            )

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info(
                    "ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.".format(
                        epoch, step, num_kg_batch
                    )
                )
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (step % args.kg_print_every) == 0:
                logging.info(
                    "KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}".format(
                        epoch,
                        step,
                        num_kg_batch,
                        time() - time4,
                        kg_batch_loss.item(),
                        kg_total_loss / step,
                    )
                )
        logging.info(
            "KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}".format(
                epoch, num_kg_batch, time() - time3, kg_total_loss / num_kg_batch
            )
        )

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.train_relation_dict.keys())
        # gc.collect()
        # torch.cuda.empty_cache()
        model(h_list, t_list, r_list, relations, mode="update_att")

        logging.info(
            "Update Attention: Epoch {:04d} | Total Time {:.1f}s".format(
                epoch, time() - time5
            )
        )

        logging.info(
            "tcm + KG Training: Epoch {:04d} | Total Time {:.1f}s".format(
                epoch, time() - time0
            )
        )

        # evaluate tcm 
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(
                model, data, testloader, Ks, device, save_dir, save=None
            )
            logging.info(
                "tcm Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}, {:.4f}]".format(
                    epoch,
                    time() - time6,
                    metrics_dict[k_min]["precision"],
                    metrics_dict[k_mid]["precision"],
                    metrics_dict[k_max]["precision"],
                    metrics_dict[k_min]["recall"],
                    metrics_dict[k_mid]["recall"],
                    metrics_dict[k_max]["recall"],
                    metrics_dict[k_min]["ndcg"],
                    metrics_dict[k_mid]["ndcg"],
                    metrics_dict[k_max]["ndcg"],
                    metrics_dict[k_min]["f1"],
                    metrics_dict[k_mid]["f1"],
                    metrics_dict[k_max]["f1"],
                )
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ["precision", "recall", "ndcg", "f1"]:
                    metrics_list[k][m].append(metrics_dict[k][m])

            #  early stopping strategy
            best_recall, should_stop = early_stopping(
                metrics_list[k_min]["recall"], args.stopping_steps
            )

            if should_stop:
                print("early stopping")
                break

            # save parameters for pretraining
            if metrics_list[k_min]["recall"].index(best_recall) == len(epoch_list) - 1:
                save_model(model, save_dir, epoch, best_epoch)
                logging.info("Save model on epoch {:04d}!".format(epoch))
                best_epoch = epoch

    # save metrics  
    metrics_df = [epoch_list]
    metrics_cols = ["epoch_idx"]
    for k in Ks:
        for m in ["precision", "recall", "ndcg", "f1"]:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append("{}@{}".format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(save_dir + "metrics.csv", sep="\t", index=False)

    # print best metrics
    best_metrics = (
        metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
    )
    logging.info(
        "Best tcm Evaluation: Epoch {:04d} | Precision [{:.4f},{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}, {:.4f}]".format(
            int(best_metrics["epoch_idx"]),
            best_metrics["precision@{}".format(k_min)],
            best_metrics["precision@{}".format(k_mid)],
            best_metrics["precision@{}".format(k_max)],
            best_metrics["recall@{}".format(k_min)],
            best_metrics["recall@{}".format(k_mid)],
            best_metrics["recall@{}".format(k_max)],
            best_metrics["ndcg@{}".format(k_min)],
            best_metrics["ndcg@{}".format(k_mid)],
            best_metrics["ndcg@{}".format(k_max)],
            best_metrics["f1@{}".format(k_min)],
            best_metrics["f1@{}".format(k_mid)],
            best_metrics["f1@{}".format(k_max)],
        )
    )

    return testloader, save_dir


def predict(args, data, testloader, save_dir):
    # GPU / CPU
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # load data
    # data = DataLoaderConsModel(args, logging)

    # load model
    model = Consecutive_visit_Model(
        args,
        data.num_entities,
        data.num_relations,
        data.num_symptoms,
        data.num_herbs,
        device,
    )
    model = load_model(model, save_dir)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_mid = Ks[1]
    k_max = max(Ks)

    tcm_scores, metrics_dict = evaluate(
        model, data, testloader, Ks, device, save_dir, save=1
    )
    np.save(save_dir + "tcm_scores.npy", tcm_scores)
    print(
        "tcm Evaluation: Precision [{:.4f}, {:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}, {:.4f}],F1 [{:.4f}, {:.4f}, {:.4f}]".format(
            metrics_dict[k_min]["precision"],
            metrics_dict[k_mid]["precision"],
            metrics_dict[k_max]["precision"],
            metrics_dict[k_min]["recall"],
            metrics_dict[k_mid]["recall"],
            metrics_dict[k_max]["recall"],
            metrics_dict[k_min]["ndcg"],
            metrics_dict[k_mid]["ndcg"],
            metrics_dict[k_max]["ndcg"],
            metrics_dict[k_min]["f1"],
            metrics_dict[k_mid]["f1"],
            metrics_dict[k_max]["f1"],
        )
    )

    # tcm_scores, metrics_dict = evaluate(model, data, testloader, Ks, device, save_dir, save=1)


if __name__ == "__main__":
    args = parse_tcm_args()
    dataloader, trainloader, testloader, save_dir = load_data(args)
    # train(args, dataloader, trainloader, testloader, save_dir)
    predict(
        args,
        dataloader,
        testloader,
        save_dir="./trained_model/TCM/lstm-gcn1/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/log2/model_epoch16.pth",
    )
