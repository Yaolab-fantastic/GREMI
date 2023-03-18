import os
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import time
import torch
import sklearn.metrics
from model_GAT import *

# Env
from utils import *

# 载入组学数据和邻接矩阵,制作数据集，其中训练集245人，测试集106人
data_tr = np.array(pd.read_csv('mogo-tr-brca.csv', header=None))[:, 1:].astype(float)
tr_omic = torch.FloatTensor(data_tr[:, 0:2503]).requires_grad_()
tr_labels = data_tr[:, 2503]
tr_labels = torch.LongTensor(tr_labels)

data_te = np.array(pd.read_csv('mogo-te-brca.csv', header=None))[:, 1:].astype(float)
te_omic = torch.FloatTensor(data_te[:, 0:2503]).requires_grad_()
te_labels = data_te[:, 2503]
te_labels = torch.LongTensor(te_labels)

adj1 = np.array(pd.read_csv('adj1.csv', header=None))[:, :].astype(float)
adj2 = np.array(pd.read_csv('adj2.csv', header=None))[:, :].astype(float)
adj3 = np.array(pd.read_csv('adj3.csv', header=None))[:, :].astype(float)

exp_adj1 = torch.LongTensor(np.where(adj1 > 0.08, 1, 0))
exp_adj2 = torch.LongTensor(np.where(adj2 > 0.08, 1, 0))
exp_adj3 = torch.LongTensor(np.where(adj3 > 0.08, 1, 0))


tr_dataset = torch.utils.data.TensorDataset(tr_omic, tr_labels)
tr_data_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=32, shuffle=True)
te_dataset = torch.utils.data.TensorDataset(te_omic, te_labels)
te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_function = nn.CrossEntropyLoss()
input_in_dim = [1000,1000,503]
input_hidden_dim = [64]
network = Fusion(num_class=5, num_views=3, hidden_dim=input_hidden_dim, dropout=0.3, in_dim=input_in_dim)


checkpoint = torch.load('model-brca.pth')
network.load_state_dict(checkpoint['net'])

network.to(device)

network.eval()

test_loss = 0.0
test_corrects = 0
test_num = 0
output_y = torch.zeros(1,5)
pred_y = torch.zeros(1)
label_y = torch.zeros(1)
#
with torch.no_grad():
    for i, data in enumerate(te_data_loader, 0):
        batch_x, targets = data
        batch_x1 = batch_x[:, 0:1000].reshape(-1, 1000, 1)
        batch_x2 = batch_x[:, 1000:2000].reshape(-1, 1000, 1)
        batch_x3 = batch_x[:, 2000:].reshape(-1, 503, 1)
        batch_x1 = batch_x1.to(torch.float32)
        batch_x2 = batch_x2.to(torch.float32)
        batch_x3 = batch_x3.to(torch.float32)
        targets = targets.long()
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_x3 = batch_x3.to(device)
        targets = targets.to(device)
        exp_adj1 = exp_adj1.to(device)
        exp_adj2 = exp_adj2.to(device)
        exp_adj3 = exp_adj3.to(device)

        te_logits = network.infer(batch_x1, batch_x2, batch_x3, exp_adj1, exp_adj2, exp_adj3)
        te_prob = F.softmax(te_logits, dim=1)
        te_pre_lab = torch.argmax(te_prob, 1)

        output_y = torch.cat((output_y, te_logits.data.cpu()), dim=0)
        pred_y = torch.cat((pred_y, te_pre_lab.data.cpu()), dim=0)
        label_y = torch.cat((label_y, targets.data.cpu()), dim=0)
        test_corrects += torch.sum(te_pre_lab == targets.data)
        test_num += batch_x1.size(0)
    real_output_y = output_y[1:, :]
    real_pred_y = pred_y[1:]
    real_label_y = label_y[1:]
    test_acc = test_corrects.double().item() / test_num

    sk_acc = sklearn.metrics.accuracy_score(real_label_y, real_pred_y)
    sk_f1marco = sklearn.metrics.f1_score(real_label_y, real_pred_y, average='macro')
    sk_f1weighted = sklearn.metrics.f1_score(real_label_y, real_pred_y, average='weighted')
    print('acc : {:.8f}'.format(sk_acc))
    print('f1_weighted : {:.8f}'.format(sk_f1weighted))
    print('f1_marco : {:.8f}'.format(sk_f1marco))
    print('end')

