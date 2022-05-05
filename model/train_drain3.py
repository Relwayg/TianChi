import pandas as pd
from sklearn.model_selection import  train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import os

from model import RNN,TrainSet

import numpy as np

a= np.array([1])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_train = pd.read_csv('data/preliminary_sel_log_dataset.csv')
data_test = pd.read_csv('data/preliminary_sel_log_dataset_a.csv')
data = pd.concat([data_train, data_test])

from drain3 import TemplateMiner  # 开源在线日志解析框架
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
config.load('./data/drain3.ini')  ## 这个文件在drain3的github仓库里有
config.profiling_enabled = False

drain_file = './data/comp_a_sellog'
persistence = FilePersistence(drain_file + '.bin')
template_miner = TemplateMiner(persistence, config=config)

##模板提取
for msg in data.msg.tolist():
    template_miner.add_log_message(msg)
temp_count = len(template_miner.drain.clusters)

## 筛选模板
template_dic = {}
size_list = []
for cluster in template_miner.drain.clusters:
    size_list.append(cluster.size)
size_list = sorted(size_list, reverse=True)[:200]  ## 筛选模板集合大小前200条，这里的筛选只是举最简单的例子。
min_size = size_list[-1]

for cluster in template_miner.drain.clusters:  ## 把符合要求的模板存下来
    print(cluster.cluster_id)
    if cluster.size >= min_size:
        template_dic[cluster.cluster_id] = cluster.size

temp_count_f = len(template_dic)


def match_template(df, template_miner, template_dic):
    msg = df.msg
    cluster = template_miner.match(msg)  # 匹配模板，由开源工具提供
    if cluster and cluster.cluster_id in template_dic:
        df['template_id'] = cluster.cluster_id  # 模板id
        df['template'] = cluster.get_template()  # 具体模板
    else:
        df['template_id'] = 'None'  # 没有匹配到模板的数据也会记录下来，之后也会用作一种特征。
        df['template'] = 'None'
    return df


data = data.apply(match_template, template_miner=template_miner, template_dic=template_dic, axis=1)
data.to_pickle(drain_file + '_result_match_data.pkl')  # 将匹配好的数据存下来

df_data = pd.read_pickle(drain_file + '_result_match_data.pkl')  # 读取匹配好模板的数据
df_data[df_data['template_id'] != 'None'].head()


def feature_generation(df_data, gap_list, model_name, log_source, win_list, func_list):
    gap_list = gap_list.split(',')

    dummy_list = set(df_data.template_id.unique())
    dummy_col = ['template_id_' + str(x) for x in dummy_list]

    for gap in gap_list:
        df_data['collect_time_gap'] = pd.to_datetime(df_data.collect_time).dt.ceil(gap)
        df_data = template_dummy(df_data)
        df_data = df_data.reset_index(drop=True)
        df_data = df_data.groupby(['sn', 'collect_time_gap']).agg(sum).reset_index()
        df_data = feature_win_fun(df_data, dummy_col, win_list, func_list, gap)
        df_data.to_pickle(
            './data/cpu_diag_comp_sel_log_all_feature_' + gap + '_' + win_list + '_' + func_list + '.pkl')  # 将构造好的特征数据存下来
        return df_data


def template_dummy(df):
    df_dummy = pd.get_dummies(df['template_id'], prefix='template_id')
    df = pd.concat([df[['sn', 'collect_time_gap']], df_dummy], axis=1)
    return df


def feature_win_fun(df, dummy_col, win_list, func_list, gap):
    win_list = win_list.split(',')
    func_list = func_list.split(',')
    drop_col = ['sn']
    merge_col = ['collect_time_gap']
    df_out = df[drop_col + merge_col]

    for win in win_list:
        for func in func_list:
            df_feature = df.groupby(drop_col).apply(rolling_funcs, win, func, dummy_col)
            df_feature = df_feature.reset_index(drop=True).rename(columns=dict(zip(dummy_col, map(lambda x: x + '_' +
                                                                                                            func + '_' + win,
                                                                                                  dummy_col))))
            df_out = pd.concat([df_out, df_feature], axis=1)
    return df_out


def rolling_funcs(df, window, func, fea_col):
    df = df.sort_values('collect_time_gap')
    df = df.set_index('collect_time_gap')
    df = df[fea_col]

    df2 = df.rolling(str(window) + 'h')

    if func in ['sum']:
        df3 = df2.apply(sum_func)
    else:
        print('func not existed')
    return df3


def sum_func(series):
    return sum(series)

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


df_data.rename(columns={'time': 'collect_time'}, inplace=True)
feature_generation(df_data, '1h', '', '', '3', 'sum')

df_data = pd.read_pickle('./data/cpu_diag_comp_sel_log_all_feature_1h_3_sum.pkl')  # 读取之前构造好的特征数据

df_train_label = pd.read_csv('data/preliminary_train_label_dataset.csv')
df_train_label_s = pd.read_csv('data/preliminary_train_label_dataset_s.csv')
df_train_label = pd.concat([df_train_label, df_train_label_s])
df_train_label = df_train_label.drop_duplicates(['sn', 'fault_time', 'label'])

df_data_train = pd.merge(df_data[df_data.sn.isin(df_train_label.sn)], df_train_label, on='sn', how='left')
y = df_data_train['label']
x = df_data_train.drop(['sn', 'collect_time_gap', 'fault_time', 'label'], axis=1)


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
rnn = RNN(n)
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
    if step % 2:
        torch.save(rnn, 'rnn.pth')
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
    print(acc.cpu().numpy(),round(value_loss,2),round(f1_value,2))
    acc_list.append(acc.cpu().numpy())
    loss_list.append(round(value_loss,2))
    f1_list.append(round(f1_value,2))
    print("Val Loss {},ACC {},F1 {}\n".format(value_loss,acc,f1_value))

    if acc > ACC:
        torch.save(rnn, 'best.pth')
        ACC = acc

acc = np.array(acc_list)
loss = np.array(loss_list)
f1_d = np.array(f1_list)
np.save('gru_acc.npy',acc)
np.save('gru_loss.npy',loss)
np.save('gru_f1.npy',f1_d)

