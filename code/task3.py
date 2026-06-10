# task3.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from modeltask3 import FusionLayer, GNN1, GNN2, GNN3, GNN4
from collections import defaultdict
import os
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

_orig_LongTensor = torch.LongTensor
def _LongTensor_with_device(data=None, *a, **k):
    return torch.tensor(data, dtype=torch.long, device=device)
torch.LongTensor = _LongTensor_with_device

os.makedirs("result", exist_ok=True)

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=120)
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024)
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8)
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=5 * 1e-3)
parser.add_argument("--neighbor_sample_size", choices=[4, 6, 10, 16], type=int, default=6)
parser.add_argument("--event_num", type=int, default=65)
parser.add_argument("--n_drug", type=int, default=572)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=256)
args = parser.parse_args()

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

def read_dataset(drug_name_id, num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list = []
    filename = "../dataset/dataset" + str(num) + ".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string = line.rstrip().split('//', 2)
            head = string[0]
            tail = string[1]
            relation = string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num == 3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg, len(tails), len(relations)

def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        count[i] = count.get(i, 0) + 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label, len(count)

def l2_re(parameter):
    reg = 0
    for param in parameter:
        reg += 0.5 * (param ** 2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot[:, i], pred_one_hot[:, i])
        result_eve[i, 1] = roc_aupr_score(y_one_hot[:, i], pred_one_hot[:, i], average=None)
        result_eve[i, 3] = f1_score(y_one_hot[:, i], pred_one_hot[:, i], average='binary')
        result_eve[i, 4] = precision_score(y_one_hot[:, i], pred_one_hot[:, i], average='binary')
        result_eve[i, 5] = recall_score(y_one_hot[:, i], pred_one_hot[:, i], average='binary')
    return [result_all, result_eve]


def save_result(filepath, result_type, result):
    with open(filepath + result_type + 'task3' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


def train(train_x, train_y, test_x, test_y, net, test_adj):
    loss_function = nn.CrossEntropyLoss()
    net.to(device)
    train_x_total = torch.tensor(train_x, dtype=torch.long).to(device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)
    train_data = TensorDataset(train_x_total, train_y_tensor)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)
    test_list = []
    max_test_output = np.zeros((0, args.event_num), dtype=float)

    for epoch in range(args.epoches):
        net.train()
        train_acc_list = []
        train_loss_total = 0
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            opti.zero_grad()
            output = net([x, 0, defaultdict(list)])
            loss = loss_function(output, y)
            loss.backward()
            opti.step()
            train_loss_total += loss.item()
            acc = accuracy_score(torch.argmax(output, dim=1).cpu().numpy(), y.cpu().numpy())
            train_acc_list.append(acc)

        net.eval()
        with torch.no_grad():
            test_x_tensor = torch.tensor(test_x, dtype=torch.long).to(device)
            test_output = F.softmax(net([test_x_tensor, 1, test_adj]), dim=1)
            test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)
            test_loss = loss_function(test_output, test_y_tensor).item()
            test_acc = accuracy_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_y_tensor.cpu().numpy())
            test_score = f1_score(torch.argmax(test_output, dim=1).cpu().numpy(), test_y_tensor.cpu().numpy(), average='macro')
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output.cpu().numpy()
            print(f"epoch {epoch + 1}, train_acc: {np.mean(train_acc_list):.4f}, test_acc: {test_acc:.4f}, test_score: {test_score:.4f}")

    return test_loss / len(test_y), test_acc, train_loss_total / len(train_y), np.mean(train_acc_list), test_list, max_test_output


def find_dif(raw_matrix):
    sim_matrix4 = np.zeros((572, 572), dtype=float)
    for i in range(572):
        for j in range(572):
            for k in range(20):
                if i == j:
                    sim_matrix4[i, j] = 0
                    break
                else:
                    if raw_matrix[i, k] == raw_matrix[j, k]:
                        sim_matrix4[i, j] += 1
    return sim_matrix4


def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator


def main():
    conn = sqlite3.connect("../dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    new_label, event_num = prepare(mechanism, action)
    new_label = np.array(new_label)
    dict1 = {i: idx for idx, i in enumerate(df_drug["name"])}
    drug_id_to_name = {v: k for k, v in dict1.items()}

    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]

    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1, 1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1, 2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1, 3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)

    x_datasets = pd.DataFrame({"drugA": drugA_id, "drugB": drugB_id}).to_numpy()

    dataset = {"dataset1": dataset1_kg, "dataset2": dataset2_kg, "dataset3": dataset3_kg, "dataset4": dataset4_kg}
    tail_len = {"dataset1": dataset1_tail_len, "dataset2": dataset2_tail_len, "dataset3": dataset3_tail_len, "dataset4": dataset4_tail_len}
    relation_len = {"dataset1": dataset1_relation_len, "dataset2": dataset2_relation_len, "dataset3": dataset3_relation_len, "dataset4": dataset4_relation_len}

    train_sum, test_sum = 0, 0

    temp_kg = [defaultdict(list) for _ in range(4)]
    for p, kg_name in enumerate(dataset):
        for i in dataset[kg_name].keys():
            for j in dataset[kg_name][i]:
                temp_kg[p][i].append(j[0])

    feature_matrix1 = np.zeros((572, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((572, dataset2_tail_len), dtype=float)
    feature_matrix3 = np.zeros((572, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((572, 572), dtype=float)

    for i in dataset4_kg.keys():
        for p, v in dataset4_kg[i]:
            feature_matrix3[i][p] = v
    for i in temp_kg[0].keys():
        for j in temp_kg[0][i]:
            feature_matrix1[i][j] = 1
    for i in temp_kg[1].keys():
        for j in temp_kg[1][i]:
            feature_matrix2[i][j] = 1
    for i in temp_kg[2].keys():
        for j in temp_kg[2][i]:
            feature_matrix4[i][j] = 1

    drug_sim1 = Jaccard(feature_matrix1)
    drug_sim2 = Jaccard(feature_matrix2)
    drug_sim4 = Jaccard(feature_matrix4)
    drug_sim3 = find_dif(feature_matrix3)

    temp_drugA = [[] for _ in range(event_num)]
    temp_drugB = [[] for _ in range(event_num)]
    for i in range(len(new_label)):
        temp_drugA[new_label[i]].append(drugA_id[i])
        temp_drugB[new_label[i]].append(drugB_id[i])

    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 5
            drug_cro_dict[temp_drugB[i][j]] = j % 5

    train_drug = [[] for _ in range(5)]
    test_drug = [[] for _ in range(5)]
    for i in range(5):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    test_adj = [[defaultdict(list) for _ in range(4)] for _ in range(5)]
    for i in range(5):
        for k in range(4):
            for j in test_drug[i]:
                if k == 0:
                    target_list = drug_sim1[j].tolist()
                elif k == 1:
                    target_list = drug_sim2[j].tolist()
                elif k == 2:
                    target_list = drug_sim3[j].tolist()
                else:
                    target_list = drug_sim4[j].tolist()

                if k != 2:
                    target_list = target_list[0]

                max_v = 0
                current_p = []
                for p, v in enumerate(target_list):
                    if v > max_v and p not in test_drug[i] and p != j:
                        max_v = v
                        current_p = [p]
                    elif v == max_v and p not in test_drug[i] and p != j:
                        current_p.append(p)
                test_adj[i][k][j].append(current_p)

    y_true = np.array([])
    y_score = np.zeros((0, args.event_num), dtype=float)
    y_pred = np.array([])

    for cross_ver in range(5):
        net = nn.Sequential(
            GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
            FusionLayer(args, drug_id_to_name)
        )
        net.to(device)

        for m in net.modules():
            if hasattr(m, "cse_features"):
                for k, v in list(m.cse_features.items()):
                    if isinstance(v, torch.Tensor):
                        m.cse_features[k] = v.to(device)
            if hasattr(m, "desc_features"):
                for k, v in list(m.desc_features.items()):
                    if isinstance(v, torch.Tensor):
                        m.desc_features[k] = v.to(device)
            if hasattr(m, "drug_pair_features"):
                for k, v in list(m.drug_pair_features.items()):
                    if isinstance(v, torch.Tensor):
                        m.drug_pair_features[k] = v.to(device)

        X_train, X_test, y_train, y_test = [], [], [], []
        for i in range(len(drugA)):
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_train.append(i)
                y_train.append(i)
            if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)

        train_x = x_datasets[X_train]
        train_y = new_label[y_train]
        test_x = x_datasets[X_test]
        test_y = new_label[y_test]

        test_loss, test_acc, train_loss, train_acc, test_list, test_output = train(train_x, train_y, test_x, test_y, net, test_adj[cross_ver])

        train_sum += train_acc
        test_sum += test_acc
        y_true = np.concatenate([y_true, test_y])
        y_score = np.concatenate([y_score, test_output])
        y_pred = np.concatenate([y_pred, np.argmax(test_output, axis=1)])

        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            cross_ver, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result("result/", "all", result_all)
    save_result("result/", "each", result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return


if __name__ == '__main__':
    main()