import pandas as pd
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import  train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import os

from model import RNN,TrainSet

import numpy as np

def f1(x,y):
    x= x.cpu().numpy()
    y = y .cpu().numpy()
    kk = pd.DataFrame()
    kk['tar'] = x 
    kk['pre'] = y
    weights =   [5/11,  4/11,  1/11,  1/11]
    macro_F1 =  0.

    for i in  range(len(weights)):
        TP =  len(kk[(kk['tar'] == i) & (kk['pre'] == i)])
        FP =  len(kk[(kk['tar'] != i) & (kk['pre'] == i)])
        FN =  len(kk[(kk['tar'] == i) & (kk['pre'] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
    
    return macro_F1

def train_all(feature_index,algo):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_data = pd.read_csv('./data/feature{}.csv'.format(feature_index))

    df_train_label = pd.read_csv('./data/preliminary_train_label_dataset.csv')
    df_train_label_s = pd.read_csv('./data/preliminary_train_label_dataset_s.csv')
    df_train_label = pd.concat([df_train_label, df_train_label_s])
    df_train_label = df_train_label.drop_duplicates(['sn', 'fault_time', 'label'])


    df_data_train = pd.merge(df_data[df_data.sn.isin(df_train_label.sn)], df_train_label, on='sn', how='left')


    y = df_data_train['label_x']
    #x = df_data_train.drop(['sn', 'collect_time_gap', 'fault_time', 'label'], axis=1)
    x = df_data_train.drop(['Unnamed: 0.1','Unnamed: 0','sn','fault_time_x','fault_time_y','label_x','label_y','fault_time_ts','server_model','event','clean_event'],axis=1)
    print(x.columns)

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=6)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    df_tensor = torch.Tensor(X_train)
    tensor_y = torch.Tensor(y_train)
    n=X_train.shape[1]
    print(n)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)
    trainset = TrainSet(df_tensor, tensor_y)
    valset = TrainSet(X_val, y_val)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    EPOCH=300
    modellr=0.0001
    ACC=0
    rnn = RNN(n,algo)
    rnn.to(device=DEVICE)
    optimizer = optim.Adam(rnn.parameters(), lr=modellr)
    loss_func=nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []
    f1_list = []
    for step in range(EPOCH):
        rnn.train()
        loss_train=0
        for tx, ty in trainloader:
            data, target = tx.to(DEVICE, non_blocking=True), ty.to(DEVICE, non_blocking=True)
            output = rnn(torch.unsqueeze(data, dim=1))
            loss = loss_func(torch.squeeze(output), target.long())
            print_loss = loss.data.item()
            loss_train+=print_loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print("epoch:",str(step),"loss:" ,str(loss_train/len(trainloader)))
        rnn.eval()
        correct = 0
        total_num = len(valloader.dataset)
        loss_val=0
        for vx, vy in valloader:
            data, target = vx.to(DEVICE, non_blocking=True), vy.to(DEVICE, non_blocking=True)
            output = rnn(torch.unsqueeze(data, dim=1))
            loss = loss_func(torch.squeeze(output), target.long())
            print_loss = loss.data.item()
            loss_val=loss_val+print_loss
            _, pred = torch.max(torch.squeeze(output), 1)
            correct += torch.sum(pred == target)

        acc = correct / total_num
        value_loss = loss_val/len(valloader)
        f1_value = f1(pred,target)
        #print(acc.cpu().numpy(),round(value_loss,2),round(f1_value,2))
        acc_list.append(acc.cpu().numpy())
        loss_list.append(value_loss)
        f1_list.append(f1_value)
        print("Val Loss {},ACC {},F1 {}\n".format(value_loss,acc,f1_value))

    acc = np.array(acc_list)
    loss = np.array(loss_list)
    f1_d = np.array(f1_list)
    np.save('./result/{}_{}_acc.npy'.format(feature_index,algo),acc)
    np.save('./result/{}_{}_loss.npy'.format(feature_index,algo),loss)
    np.save('./result/{}_{}_f1.npy'.format(feature_index,algo),f1_d)

if __name__=='__main__':
    feature_num = [1,2,3]
    algo = ['rnn','gru','lstm']

    for f_n in feature_num:
        for al in algo:
            print("model {}, feature {} ".format(al,f_n))
            train_all(f_n,al)