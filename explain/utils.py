import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import os
import logging
import time
import csv
import codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import sklearn.metrics
import torch_geometric
from scipy.sparse import coo_matrix  # 转化成COO格式
# import lifelines
# from lifelines.utils import concordance_index
# from lifelines.statistics import logrank_test

from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer

################
# Data Utils
################

def load_csv_data(k, opt):
    folder_path = './example_data/input_features_labels/split'
    print("Loading data from:", folder_path+str(k))
    train_data_path = folder_path+str(k)+'_train_320d_features_labels.csv'
    train_data = np.array(pd.read_csv(train_data_path, header=None))[1:, 2:].astype(float)

    tr_features = torch.FloatTensor(train_data[:, :320].reshape(-1, 320, 1)).requires_grad_()
    tr_labels = torch.LongTensor(train_data[:, 320:])
    print("Training features and labels:", tr_features.shape, tr_labels.shape)

    test_data_path = folder_path+str(k)+'_test_320d_features_labels.csv'
    test_data = np.array(pd.read_csv(test_data_path, header=None))[1:, 2:].astype(float)

    te_features = torch.FloatTensor(test_data[:, :320].reshape(-1, 320, 1)).requires_grad_()
    te_labels = torch.LongTensor(test_data[:, 320:])
    print("Testing features and labels:", te_features.shape, te_labels.shape)

    similarity_matrix = np.array(pd.read_csv(
        './example_data/input_adjacency_matrix/split'+str(k)+'_adjacency_matrix.csv')).astype(float)
    adj_matrix = torch.LongTensor(np.where(similarity_matrix > opt.adj_thresh, 1, 0))
    print("Adjacency matrix:", adj_matrix.shape)
    print("Number of edges:", adj_matrix.sum())

    if opt.task == "grad":
        tr_idx = tr_labels[:, 2] >= 0
        tr_labels = tr_labels[tr_idx]
        tr_features = tr_features[tr_idx]
        print("Training features and grade labels after deleting NA labels:", tr_features.shape, tr_labels.shape)

        te_idx = te_labels[:, 2] >= 0
        te_labels = te_labels[te_idx]
        te_features = te_features[te_idx]
        print("Testing features and grade labels after deleting NA labels:", te_features.shape, te_labels.shape)

    return tr_features, tr_labels, te_features, te_labels, adj_matrix


################
# Grading Utils
################
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def print_model(model, optimizer):
    print(model)
    print("Model's state_dict:")
    # Print model's state_dict
    for param_tensor in model.state_dict():
        print(param_tensor,"\t", model.state_dict()[param_tensor].size())
    print("optimizer's state_dict:")
    # Print optimizer's state_dict
    for var_name in optimizer.state_dict():
        print(var_name,"\t", optimizer.state_dict()[var_name])


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()



def compute_ROC_AUC(test_pred, gt_labels):

    enc = LabelBinarizer()
    enc.fit(gt_labels)
    labels_oh = enc.transform(gt_labels)  ## convert to one_hot grade labels.
    # print(gt_labels, labels_oh, test_pred.shape)
    fpr, tpr, thresh = roc_curve(labels_oh.ravel(), test_pred.ravel())
    aucroc = auc(fpr, tpr)

    return aucroc

def compute_metrics(test_pred, gt_labels):

    enc = LabelBinarizer()
    enc.fit(gt_labels)
    labels_oh = enc.transform(gt_labels)  ## convert to one_hot grade labels.

    # print(gt_labels, labels_oh, test_pred.shape)
    # print(labels_oh, test_pred)
    idx = np.argmax(test_pred, axis=1)
    # print(gt_labels, idx)
    labels_and_pred = np.concatenate((gt_labels, idx))
    test_pred = enc.fit(labels_and_pred).transform(labels_and_pred)[gt_labels.shape[0]:, :]
    # print(test_pred)
    macro_f1_score = f1_score(labels_oh, test_pred, average='macro')
    # micro_f1_score = f1_score(labels_oh, test_pred, average='micro') #equal to accuracy.
    precision = precision_score(labels_oh, test_pred, average='macro')
    recall = recall_score(labels_oh, test_pred, average='macro')
    # kappa = cohen_kappa_score(labels_oh, test_pred)

    return macro_f1_score, precision, recall



################
# Layer Utils
################
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

################
# subgraph by luo
################
def adj_to_PyG_edge_index(adj):
    coo_A = coo_matrix(adj)
    edge_index, edge_weight = torch_geometric.utils.convert.from_scipy_sparse_matrix(coo_A)
    return edge_index

def data_to_PyG_data(x, edge_index, y):
    out_data = x
    out_edge_index = edge_index
    out_label = y
    PyG_data = torch_geometric.data.Data(x=out_data, edge_index=out_edge_index, y=out_label)
    return PyG_data

def PyG_edge_index_to_adj(edge_index):
    adj = torch_geometric.utils.to_dense_adj(edge_index=edge_index)
    return adj

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'w+','utf-8')#追加写入方式
  writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
  for data in datas:
    writer.writerow(data)
  print("文件保存成功")