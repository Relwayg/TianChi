import pandas as pd
import torch
import numpy as np
from torch import nn
from model import RNN


def f1(x,y):
    kk = pd.DataFrame()
    kk['tar'] = x 
    kk['pre'] = y
    weights =  [3/7,  2/7,  1/7,  1/7]
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

drain_file = 'comp_a_sellog'
df_data = pd.read_pickle('/home/liuchi/wh/data_mining/大作业/data/cpu_diag_comp_sel_log_all_feature_1h_3_sum.pkl')  # 读取之前构造好的特征数据
df_test_df = pd.read_csv('/home/liuchi/wh/data_mining/大作业/data/preliminary_submit_dataset_a.csv', index_col=0).reset_index()
df_test = pd.merge(df_data[df_data.sn.isin(df_test_df.sn)], df_test_df, on='sn', how='left')
res = df_test[['sn', 'fault_time']]
print(df_test)
x_test = df_test.drop(['sn', 'collect_time_gap', 'fault_time'], axis=1)
model=torch.load('/home/liuchi/wh/data_mining/大作业/data/best.pth')
model.cpu()
x_test = np.array(x_test)

df_tensor = torch.Tensor(x_test)
output= model(torch.unsqueeze(df_tensor, dim=1))
_, pred = torch.max(torch.squeeze(output), 1)


res['label']=pred
res = res.sort_values(['sn', 'fault_time'])
res = res.drop_duplicates(['sn', 'fault_time'], keep='last')
res.to_csv('comp_a_result_1.csv', index=0)
