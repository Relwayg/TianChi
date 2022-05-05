import pandas as pd
import re
from tqdm import tqdm
from drain3 import TemplateMiner #开源在线日志解析框架
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
import gensim
import numpy as np

def mkCat(data):
    cat=[]
    for ce in tqdm(data.clean_event.tolist()):
        if ce == 'None':
            continue
        cat.extend(ce.split())
    cat = list(set(cat))
    return cat

def mkFeature(data:pd.DataFrame,cat):

    features = []
    for idx,row in tqdm(data.iterrows()):
        f = [0]*len(cat)
        ee = str(row.clean_event)
        for c in cat:
            if c in ee:
                # print(ee)
                cat_idx = cat.index(c)
                f[cat_idx]= 1 + f[cat_idx]
        # print(row.sn)
        features.append(f)
    ft = pd.DataFrame(features)
    ft.columns = cat
    data = pd.concat([data,ft],axis=1)
    return data

def mkFeature2(data:pd.DataFrame):
    # 对sel数据进行模版匹配，相当于进行了模版分类
    config = TemplateMinerConfig()
    config.load('./feature/drain3.ini') ## 这个文件在drain3的github仓库里有
    config.profiling_enabled = False

    drain_file = 'comp_a_sellog'
    persistence = FilePersistence(drain_file + '.bin')
    template_miner = TemplateMiner(persistence, config=config)

    events = data['event'].tolist()
    sel = []
    for e in events:
        ee = eval(e)
        sel.extend(list(ee))
    ##模板提取
    for msg in sel:
        # print(msg)
        template_miner.add_log_message(msg)
    print('模板数量 ',len(template_miner.drain.clusters))
    print(template_miner.drain.clusters)
    #匹配模版计算特征  
    features = []
    for idx,row in tqdm(data.iterrows()):
        f = [0]*len(template_miner.drain.clusters)
        events = eval(row.event)
        for e in events:
            c_id = template_miner.match(e).cluster_id
            # print(c_id)
            f[c_id-1] = f[c_id-1] + 1
        features.append(f)
    ft = pd.DataFrame(features)
    data = pd.concat([data,ft],axis=1)
    return data

def mkFeature3(data:pd.DataFrame):
    sentences= [] 
    events = data['event'].tolist()

    for e in events:
        ee = eval(e)
        for eee in ee:
            tmp = re_pat(eee)
            sentences.append(tmp)
    w2v_model = gensim.models.Word2Vec(sentences, min_count=5,vector_size=200, window=3, sg=0, hs=1, seed=2022) 

    features = []
    for idx,row in tqdm(data.iterrows()):
        event = row.clean_event
        vec = []
        for w in event.split():
            if w in w2v_model.wv:
                vec.append(w2v_model.wv[w])
        if len(vec)>0:
            f = np.mean(vec,axis=0)
        else:
            f = [0] * w2v_model.vector_size
        features.append(f)
    ft = pd.DataFrame(features)
    data = pd.concat([data,ft],axis=1)
    return data


def re_pat(i):
    # print(i)
    # 同类型补齐
    i = re.sub('Subsys Health','Subsystem Health',i)
    i = re.sub('_Stat ','_Status ',i)
    i = re.sub('_DIM ','_DIMM ',i)
    i = re.sub('Fully','Full',i)
    i = re.sub('BIOS_PostStatus','BIOS_POST_Status',i)
    i = re.sub('_UP','_Up',i)
    i = re.sub('ACPI_PWR_Status','ACPI_Pwr_Status',i)
    i = re.sub('Chassis_control','Chassis_Control',i)
    
    # 去除符号
    i = re.sub('[();:|,#/]','',i)
    i = re.sub(r'�Tota','',i)
    i = re.sub(r'�','',i)

    # 相同合并
    i = re.sub(r'@DIMM[12349AB(@&#9;/0]+CPU[12]','DIMMCPU',i)
    i = re.sub(r'DIMM[0-9]+','DIMM',i)
    i = re.sub(r'DIMMG[0123]','DIMMG',i)
    
    i = re.sub(r'CPU[0123A-F]+','CPU',i)
    
    i = re.sub(r'BP[1-4]+_','BP_',i)

    i = re.sub(r'DISK[0-9]+','DISK',i)

    i = re.sub(r'FAN[0-9]+','FAN',i)
    i = re.sub(r'FAN_[0-9]+','FAN',i)

    i = re.sub(r'Front[0-9]+_[0-9]+_Status','Front_Status',i)

    i = re.sub(r'L_[0-9]+','L',i)
    i = re.sub(r'R_[0-9]+','R',i)
    
    i = re.sub(r'R[123]_','R_',i)
    i = re.sub(r'Cable_[0-9]+','Cable',i)

    i = re.sub(r'MEM_CH[A-H01]+','MEM_CH',i)

    i = re.sub(r'HDD[0-9]+','HDD',i)

    i = re.sub(r'NVME[0-9]+','NVME',i)
    i = re.sub(r'NVMeSSD_[0-9]','NVMeSSD',i)

    i = re.sub(r'OEM_PS[01]','OEM_PS',i)

    i = re.sub(r'Rear[2]_','Rear',i)
    i = re.sub(r'Rear[0-9]_Status','Rear_Status',i)

    i = re.sub(r'Riser[0-9]','Riser',i)

    i = re.sub(r'MB_SSD[0-9]','MB_SSD',i)

    i = re.sub(r'Watchdog2','Watchdog',i)

    i = re.sub(r'PS[0-9]+','PS',i)
    i = re.sub(r'PSU[0-9]+','PSU',i)

    i = re.sub(r'Eth[0-9]','Eth',i)

    i = re.sub(r'Port[0-9]','Port',i)

    i = re.sub(r'FPGA[0-9]+_','FPGA ',i)

    i = re.sub(r'[abcdef0-9]+  OEM record ef  aa[0-9a-z]+','OEM record ef aa',i)

    # 匹配十六进制
    i = re.sub(r'0x[0-9a-f]{2}','HEX',i)
    # i = re.sub( r'aa[1-9a-f]+','memorynum', i)

    # 匹配纯数字
    i = re.sub(r' [0-9]+',' ',i)
    i = re.sub(r' -[0-9]+',' ',i)
    # i = re.sub(r' [0-9a-z]{12}',' ',i)
    # i = re.sub(r' [0-9a-z]{6}',' ',i)

    # 删除其他
    i = re.sub(r' - ',' ',i)

    # 处理_连词
    i = re.sub(r'ACPI_Pwr_Status','ACPI Pwr Status',i)
    i = re.sub(r'Power_Status','Power Status',i)
    i = re.sub(r'CPU_DIMM_HOT','CPU DIMM HOT',i)
    i = re.sub(r'IPMI_Watchdog','IPMI Watchdog',i)
    i = re.sub(r'Event_Log','EventLog',i)
    i = re.sub(r'PS_Status','PS Status',i)
    i = re.sub(r'Power_Limiting','Power Limiting',i)
    i = re.sub(r'System_Restart','System Restart',i)

    # 处理连词
    i = re.sub(r'PROCHOT','PROC HOT',i)
    i = re.sub(r'PowerLimiting','Power Limiting',i)
    i = re.sub(r'SlotConnector','Slot Connector',i)
    i = re.sub(r'MicrocontrollerCoprocessor','Microcontroller Coprocessor',i)
    
    # 删除错误词
    i = re.sub(r'SysRestart','',i)
    i = re.sub(r' Syst ','',i)

    # 处理OEM
    if 'OEM record' in i:
        i = i[:14]
    i = re.sub(r' +',' ',i)
    return i

def countWord(msgs):
    contW = {}
    for m in msgs:
        m_re = re_pat(m)
        for mm in m_re.split():
            if mm not in contW.keys():
                contW[mm]=1
            else:
                contW[mm] = contW[mm]+1

    return contW

def cleanEvent(df):
    cleanE = []

    for idx,row in df.iterrows():
        c = ''
        if row.event == 'None':
            c = ''
        else:
            # print(row.sn,' ',row.event)
            e = eval(str(row.event))
            for ee in e:
                eee = re_pat(ee)
                # print(ee)
                # print(eee)
                c = c + ' ' + eee
            # print(c)
        cleanE.append(c)
    
    df['clean_event'] = cleanE
    return df


# data = pd.read_csv('train_data.csv')
# data = data[data.event!='None']
# data_a = pd.read_csv('test_data_a.csv')
# data_a = data_a[data_a.event!='None']

# cat = mkCat(data,data_a)
# print(cat)

# data_mkfeature = mkFeature(data,cat)
# data_mkfeature.to_csv('train_data_feature.csv')

# data_mkfeature_a = mkFeature(data_a,cat)
# data_mkfeature_a.to_csv('test_data_feature_a.csv')


                
