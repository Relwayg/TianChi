# 第三届阿里云磐久智维算法大赛

### 1. 说明
该文件是第三届阿里云磐久智维算法大赛赛题解题代码，也是北理工数据挖掘课程项目大作业

其中包含了两种特征提取方法和两种模型方法，项目具体报告内容参见报告文件夹

当前代码排名：**81/1758**

### 2. 目录结构

````
|-- project
    |--code
        |-- main.py
    |--data
        |-- preliminary_a_test
            |-- final_sel_log_dataset_a.csv
            |-- final_submit_dataset_a.csv
        |-- preliminary_b_test
            |-- preliminary_sel_log_dataset_b.csv
            |-- preliminary_submit_dataset_b.csv
        |-- preliminary_train
            |-- additional_sel_log_dataset.csv
            |-- preliminary_crashdump_dataset.csv
            |-- preliminary_sel_log_dataset.csv
            |-- preliminary_train_label_dataset_s.csv
            |-- preliminary_train_label_dataset.csv
            |-- preliminary_venus_dataset.csv
    |-- featrure
        |-- mkdta.py
        |-- mkfeature.py
    |-- model
        |-- catboost.py
        |-- draw.py
        |-- model.py
        |-- test.py
        |-- test.py
        |-- train_train3.py
        |-- train_feature.py
    |-- user_data
        |-- train_data.csv
        |-- train_data_featue.csv
    |-- result
    |-- 报告
    |-- 数据挖掘.pptx
  
````
- code：存放主程序
- data：存放原始数据
- feature：存放特征提取，数据构建代码
- model：存放模型代码
- user_data：存放程序中间过程产生的代码
- 报告：所有的算法细节报告文件
- 数据挖掘：项目ppt

### 3. 运行方式
直接在命令行运行 ```` sh run.sh ```` 即可

### 4. 所需库
````
catboost==1.0.5
drain3==0.9.9
gensim==4.1.2
pandas==1.2.3
tqdm==4.62.3
numpy==1.21.3
````

### 5. 相关网址
**baseline1** 
https://github.com/LogicJake/competition_baselines/tree/master/competitions/tianchi_aiops2022

**baseline2**
https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.164e36fbtbc52x&postId=345786

**baseline3**
https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/123612898

**复赛例子**
https://tianchi.aliyun.com/forum/postDetail?postId=366798

**catboost**\
https://blog.csdn.net/weixin_42305672/article/details/111252715
https://catboost.ai/en/docs/features/visualization_jupyter-notebook