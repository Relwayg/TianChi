from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import os

import sys
sys.path.append('.')
from model.catboost import macro_f1, run_ctb
from feature.mkdata import mkData
from feature.mkfeature import cleanEvent, mkCat, mkFeature, mkFeature2, mkFeature3
# from feature.mkfeature import cleanEvent, mkCat, mkFeature
# 构建日志事件

sel_train_0 = pd.read_csv('./data/preliminary_train/additional_sel_log_dataset.csv')
sel_train_1 = pd.read_csv('./data/preliminary_train/preliminary_sel_log_dataset.csv')
label_train_0 = pd.read_csv('./data/preliminary_train/preliminary_train_label_dataset_s.csv')
label_train_1 = pd.read_csv('./data/preliminary_train/preliminary_train_label_dataset.csv')
crashdump_train = pd.read_csv('./data/preliminary_train/preliminary_crashdump_dataset.csv')
venus_train = pd.read_csv('./data/preliminary_train/preliminary_venus_dataset.csv')

sel_test_a = pd.read_csv('./data/preliminary_a_test/final_sel_log_dataset_a.csv')
lab_test_a = pd.read_csv('./data/preliminary_a_test/final_submit_dataset_a.csv')

# sel_test_a = pd.read_csv('tcdata/final_sel_log_dataset_a.csv')
# lab_test_a = pd.read_csv('tcdata/final_submit_dataset_a.csv')

# sel_test_b = pd.read_csv('./data/preliminary_b_test/preliminary_sel_log_dataset_b.csv')
# lab_test_b = pd.read_csv('./data/preliminary_b_test/preliminary_submit_dataset_b.csv')

label_train = pd.concat([label_train_0,label_train_1],axis=0).drop_duplicates(['sn','fault_time']).sort_values(by=['sn','fault_time'],ignore_index=True)
sel_train = pd.concat([sel_train_0,sel_train_1],axis=0).drop_duplicates(['sn','time']).sort_values(by=['sn','time'],ignore_index=True)

print('读取数据完毕')

print('开始构建训练数据')
if os.path.exists('./user_data/train_data.csv'):
    train_data = pd.read_csv('./user_data/train_data.csv')
else:
    train_data,err_idx,err_sn = mkData(sel_train,label_train)
    train_data.to_csv('./user_data/train_data.csv')
    err = pd.DataFrame([[err_idx,err_sn]],columns=['err_idx','err_sn'])
    err.to_csv('./user_data/err.csv')

print('开始构建测试A数据')
test_data_a,err_idx_a,err_sn_a = mkData(sel_test_a,lab_test_a)
test_data_a.to_csv('./user_data/test_data_a.csv')
err_a = pd.DataFrame([[err_idx_a,err_sn_a]],columns=['err_idx_a','err_sn_a'])
err_a.to_csv('./user_data/err_a.csv')

# print('开始构建测试B数据')
# test_b,err_idx_b,err_sn_b = mkData(sel_test_b,lab_test_b)
# test_b.to_csv('./user_data/test_data_b.csv')
# err_b = pd.DataFrame([[err_idx_b,err_sn_b]],columns=['err_idx_b','err_sn_b'])
# err_b.to_csv('./user_data/err_b.csv')

print('数据构建完毕，开始构建特征')

print('对数据开始清洗')
train_data = train_data[train_data.event != 'None'].reset_index(drop=True)
train_data_1 = cleanEvent(train_data)
test_data_a_1 = cleanEvent(test_data_a)
# for idx,row in train_data_1.iterrows():
#     print(row.sn)
#     print(row.event)
#     print(row.clean_event)

print('训练数据构建特征')
cat = mkCat(train_data_1)
print(cat)

print('开始生成特征feature')
train_data_2 = mkFeature(train_data_1,cat)
test_data_a_2 = mkFeature(test_data_a_1,cat)
# train_data_2 = mkFeature2(train_data_1)
# test_data_a_2 = mkFeature2(test_data_a_1)
# train_data_2 = mkFeature3(train_data_1)
# test_data_a_2 = mkFeature3(test_data_a_1)
train_data_2.to_csv('./user_data/feature.csv')
# train_data_2.to_csv('jsee.csv')
# print(train_data_2[['event']].head)
# print('')

# print(train_data_2.columns)

print('开始训练测试阶段')
train = train_data_2
test = test_data_a_2
classes = np.unique(train['label'])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train['label'])
class_weights = dict(zip(classes, weights))
print('标签权重设置为',class_weights)

NUM_CLASSES = train['label'].nunique()
print(NUM_CLASSES)
FOLDS = 10
TARGET = 'label'
use_features = [col for col in train.columns if col not in ['sn','fault_time','Unnamed: 0','fault_time_ts','clean_event','event',TARGET]]
print('特征为',use_features)

print('开始10轮交叉训练')
test_pre,train_pre =  run_ctb(train, test, use_features,TARGET,NUM_CLASSES,FOLDS,class_weights)

print(train.columns)
tar = train[['sn', 'label', 'fault_time']].copy()
pre = tar.copy()
pre['label'] = train_pre.argmax(axis=1)

print('开始计算训练集合加权F1')
F1 = macro_f1(tar,pre)

print('加权F1: ',F1)

print('开始保存测试集合结果')
sub = test[['sn','fault_time']].copy()
sub['label'] = test_pre.argmax(axis=1)
sub.to_csv('./prediction_result/final_pred_a.csv',index=False)

print('OVER OVER OVER!!!')

