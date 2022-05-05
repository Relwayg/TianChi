import pandas as pd 
import numpy as np
from tqdm import tqdm

def mkData(sel:pd.DataFrame,lab:pd.DataFrame):
    sel = sel.drop_duplicates(['sn','time']).sort_values(by=['sn','time'],ignore_index=True)
    sel = sel[sel.sn.isin(lab.sn)]
    lab = lab.drop_duplicates(['sn','fault_time']).sort_values(by=['sn','fault_time'],ignore_index=True)

    sel['time'] = pd.to_datetime(sel['time'])
    lab['fault_time'] = pd.to_datetime(lab['fault_time'])
    sel['time_ts'] = sel["time"].values.astype(np.int64) // 10 ** 9
    lab['fault_time_ts'] = lab["fault_time"].values.astype(np.int64) // 10 ** 9

    g_lab = lab.groupby(by='sn',as_index=False).agg(list)

    events = []
    server_models = []
    err_idx = []
    err_sn = []

    for idx,row in tqdm(g_lab.iterrows()):
        sn = row.sn
        ft = row.fault_time
        ftt = row.fault_time_ts
        sm = sel[sel['sn']==sn]['server_model'].tolist()[0]

        if len(ft)==1: #说明这个服务器只有一个报错
            events.append(set(sel[sel['sn']==sn]['msg'].tolist()))
            server_models.append(sm)
        else:
            for i in range(len(ft)):
                if i==0: #说明是开始只需要找到小于fulat_time[i]时间的事件信息
                    eve = set(sel[(sel['sn']==sn)&(sel['time']<=ft[i])]['msg'].tolist())
                else: #说明第二个开始，这时候需要找到 fulat_time[i-1]<time<=fulat_time[i]
                    eve = set(sel[(sel['sn']==sn )&(sel['time']<=ft[i]) & (sel['time']>ft[i-1]) ]['msg'].tolist())
                
                if len(eve)==0:#说明不存在事件，会有下面可能
                    if i==0: #说明第一个是空的，是干扰数据
                        eve = 'None'
                        print('None')
                        err_idx.append(idx)
                        err_sn.append(sn)
                    else:#说明可能是间隔太小
                        if len(ft)==2: #说明是两个,则直接让其小于ft[i]即可
                            eve = set(sel[(sel['sn']==sn )&(sel['time']<=ft[i])]['msg'].tolist())
                        else:
                            for k in range(2,i):
                                eve = set(sel[(sel['sn']==sn )&(sel['time']<=ft[i]) & (sel['time']>ft[i-k]) ]['msg'].tolist())
                                if len(eve)!=1:
                                    break

                if len(eve)==0:
                    eve = set(sel[(sel['sn']==sn )&(sel['time_ts']<=ftt[i])&(sel['time_ts']>(ftt[i]-48*60*60))]['msg'].tolist())

                events.append(eve)
                server_models.append(sm)


    lab['server_model'] = server_models
    lab['event'] = events
    return lab,err_idx,err_sn


# sel_train_0 = pd.read_csv('./data/preliminary_train/additional_sel_log_dataset.csv')
# sel_train_1 = pd.read_csv('./data/preliminary_train/preliminary_sel_log_dataset.csv')
# label_train_0 = pd.read_csv('./data/preliminary_train/preliminary_train_label_dataset_s.csv')
# label_train_1 = pd.read_csv('./data/preliminary_train/preliminary_train_label_dataset.csv')
# crashdump_train = pd.read_csv('./data/preliminary_train/preliminary_crashdump_dataset.csv')
# venus_train = pd.read_csv('./data/preliminary_train/preliminary_venus_dataset.csv')

# label_train = pd.concat([label_train_0,label_train_1],axis=0).drop_duplicates(['sn','fault_time']).sort_values(by=['sn','fault_time'],ignore_index=True)
# sel_train = pd.concat([sel_train_0,sel_train_1],axis=0).drop_duplicates(['sn','time']).sort_values(by=['sn','time'],ignore_index=True)

# train_data,err_idx,err_sn = mkData(sel_train,label_train)
# train_data.to_csv('train_data.csv')
# err = pd.DataFrame([[err_idx,err_sn]],columns=['err_idx','err_sn'])
# err.to_csv('err.csv')

# sel_test_a = pd.read_csv('./data/preliminary_a_test/preliminary_sel_log_dataset_a.csv')
# lab_test_a = pd.read_csv('./data/preliminary_a_test/preliminary_submit_dataset_a.csv')

# test_a,err_idx_a,err_sn_a = mkData(sel_test_a,lab_test_a)
# test_a.to_csv('test_data_a.csv')
# err_a = pd.DataFrame([[err_idx_a,err_sn_a]],columns=['err_idx_a','err_sn_a'])
# err_a.to_csv('err_a.csv')

# sel_test_b = pd.read_csv('./data/preliminary_b_test/preliminary_sel_log_dataset_b.csv')
# lab_test_b = pd.read_csv('./data/preliminary_b_test/preliminary_submit_dataset_b.csv')

# test_b,err_idx_b,err_sn_b = mkData(sel_test_b,lab_test_b)
# test_b.to_csv('test_data_b.csv')
# err_b = pd.DataFrame([[err_idx_b,err_sn_b]],columns=['err_idx_b','err_sn_b'])
# err_b.to_csv('err_b.csv')