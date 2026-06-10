# MF.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches", type=int, choices=[100, 500, 1000, 2000], default=120)
parser.add_argument("--batch_size", type=int, choices=[2048, 1024, 512, 256, 128], default=1024)
parser.add_argument("--weigh_decay", type=float, choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-8], default=1e-8)
parser.add_argument("--lr", type=float, choices=[1e-3, 1e-4, 1e-5, 4 * 1e-3], default=1 * 1e-2)
parser.add_argument("--neighbor_sample_size", choices=[4, 6, 10, 16], type=int, default=6)
parser.add_argument("--event_num", type=int, default=65)
parser.add_argument("--n_drug", type=int, default=572)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--embedding_num", type=int, choices=[128, 64, 256, 32], default=128)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

_orig_LongTensor = torch.LongTensor
def _LongTensor_with_device(data=None, *a, **k):
    return torch.tensor(data, dtype=torch.long, device=device)
torch.LongTensor = _LongTensor_with_device

os.makedirs("../result", exist_ok=True)

from modeltask1 import FusionLayerWithCSE, GNN1, GNN2, GNN3, GNN4

def setup_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label

def l2_re(parameter):
    reg = 0
    for param in parameter:
        reg += 0.5 * (param ** 2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
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
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(), average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(), average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(), average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(), average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(), average='binary')
    return [result_all, result_eve]

def save_result(filepath, result_type, result):
    with open(filepath + result_type + 'task1' + '.csv', "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def train(train_x, train_y, test_x, test_y, net):
    loss_function = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weigh_decay)

    train_losses = []
    test_losses = []
    test_accs = []
    test_f1s = []
    train_accs = []

    best_test_acc = 0

    train_x1 = train_x.copy()
    train_x2 = train_x.copy()
    train_x2[:, [0, 1]] = train_x2[:, [1, 0]]
    train_x_total = torch.tensor(np.concatenate([train_x1, train_x2], axis=0), dtype=torch.long)
    train_y_tensor = torch.tensor(np.concatenate([train_y, train_y]), dtype=torch.long)
    train_data = TensorDataset(train_x_total, train_y_tensor)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)

    max_test_output = np.zeros((0, args.event_num), dtype=float)

    for epoch in range(args.epoches):
        net.train()
        epoch_train_loss = 0.0
        batch_train_accs = []

        for x_batch, y_batch in train_iter:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            opti.zero_grad()
            logits, _ = net(x_batch)
            loss = loss_function(logits, y_batch)
            loss.backward()
            opti.step()

            epoch_train_loss += loss.item()
            preds_batch = torch.argmax(logits, dim=1).cpu().numpy()
            batch_train_accs.append(accuracy_score(preds_batch, y_batch.cpu().numpy()))

        avg_train_loss = epoch_train_loss / (len(train_iter) if len(train_iter) > 0 else 1)
        avg_train_acc = np.mean(batch_train_accs) if len(batch_train_accs) > 0 else 0.0
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        net.eval()
        with torch.no_grad():
            test_x_tensor = torch.tensor(test_x, dtype=torch.long).to(device)
            test_label = torch.tensor(test_y, dtype=torch.long).to(device)
            test_logits, _ = net(test_x_tensor)
            test_probs = F.softmax(test_logits, dim=1)
            test_loss_val = loss_function(test_logits, test_label).item()
            test_preds = torch.argmax(test_probs, dim=1).cpu().numpy()
            test_acc_val = accuracy_score(test_preds, test_label.cpu().numpy())
            test_f1_val = f1_score(test_preds, test_label.cpu().numpy(), average='macro')

        test_losses.append(test_loss_val)
        test_accs.append(test_acc_val)
        test_f1s.append(test_f1_val)

        if test_acc_val > best_test_acc:
            best_test_acc = test_acc_val

        probs_np = test_probs.cpu().numpy()

        if test_f1s[-1] == max(test_f1s):
            max_test_output = probs_np

        print('epoch [%d/%d] train_loss: %.6f test_loss: %.6f train_acc: %.4f test_acc: %.4f test_f1: %.4f' %
              (epoch + 1, args.epoches, avg_train_loss, test_loss_val, avg_train_acc, test_acc_val, test_f1_val))

    return (
        np.mean(test_losses),
        best_test_acc,
        np.mean(train_losses),
        np.mean(train_accs),
        test_f1s,
        max_test_output,
        train_losses,
        test_losses,
        test_accs,
        test_f1s
    )

def main():
    conn = sqlite3.connect("../dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    new_label = prepare(mechanism, action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)

    drug_id_to_name = {v: k for k, v in dict1.items()}

    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1, 1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1, 2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1, 3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets).to_numpy()
    dataset = {"dataset1": dataset1_kg, "dataset2": dataset2_kg, "dataset3": dataset3_kg, "dataset4": dataset4_kg}
    tail_len = {"dataset1": dataset1_tail_len, "dataset2": dataset2_tail_len, "dataset3": dataset3_tail_len, "dataset4": dataset4_tail_len}
    relation_len = {"dataset1": dataset1_relation_len, "dataset2": dataset2_relation_len, "dataset3": dataset3_relation_len, "dataset4": dataset4_relation_len}
    train_sum, test_sum = 0, 0

    y_true = np.array([])
    y_score = np.zeros((0, args.event_num), dtype=float)
    y_pred = np.array([])

    all_train_losses = []
    all_test_losses = []
    all_test_accs = []
    all_test_f1s = []

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    kfold = kf.split(x_datasets, new_label)
    for i, (train_idx, test_idx) in enumerate(kfold):
        print(f"===== Fold {i} =====")
        net = nn.Sequential(
            GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
            GNN4(dataset, tail_len, relation_len, args, dict1, drug_name),
            FusionLayerWithCSE(args, drug_id_to_name=drug_id_to_name)
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

        train_x = x_datasets[train_idx]
        train_y = new_label[train_idx]
        test_x = x_datasets[test_idx]
        test_y = new_label[test_idx]

        returned = train(train_x, train_y, test_x, test_y, net)
        test_loss, test_acc, train_loss, train_acc, test_list, test_output = returned[:6]
        train_losses = returned[6]
        test_losses = returned[7]
        test_accs = returned[8]
        test_f1s = returned[9]

        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_test_accs.append(test_accs)
        all_test_f1s.append(test_f1s)

        train_sum += train_acc
        test_sum += test_acc

        pred_type = np.argmax(test_output, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, test_output))
        y_true = np.hstack((y_true, test_y))
        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            i, test_loss, test_acc, train_loss, train_acc))

    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result("../result/", "all", result_all)
    save_result("../result/", "each", result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    setup_seed()
    main()
