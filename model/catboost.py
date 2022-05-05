from cProfile import label
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from catboost import CatBoostClassifier, Pool
import gc
import numpy as np
import pandas as pd


# train = pd.read_csv('user_data/train_data_feature.csv')
# classes = np.unique(train['label'])
# weights = compute_class_weight(class_weight='balanced', classes=classes, y=train['label'])
# class_weights = dict(zip(classes, weights))
# print(class_weights)

# NUM_CLASSES = train['label'].nunique()
# FOLDS = 10
# TARGET = 'label'
# use_features = [col for col in train.columns if col not in ['sn','fault_time','Unnamed: 0','fault_time_ts',TARGET]]

def run_ctb(df_train, df_test, use_features,TARGET,NUM_CLASSES,FOLDS,class_weights):
    target = TARGET
    oof_pred = np.zeros((len(df_train), NUM_CLASSES))
    y_pred = np.zeros((len(df_test), NUM_CLASSES))
    
    folds = GroupKFold(n_splits=FOLDS)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(df_train, df_train[TARGET], df_train['sn'])):
        print(f'Fold {fold + 1}') 
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind] 
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        
        train_dataset = Pool(x_train,y_train,cat_features=[0])
        val_dataset = Pool(x_val,y_val,cat_features=[0])
        test_dataset = Pool(df_test[use_features],cat_features=[0])

        params = { 
            'task_type': 'CPU', 
            'bootstrap_type': 'Bernoulli',
            'learning_rate': 0.02, 
            'eval_metric': 'MultiClass', 
            'loss_function': 'MultiClass', 
            'classes_count': NUM_CLASSES, 
            'iterations': 500, 
            'random_seed': 2022, 
            'depth': 10, 
            'subsample': 0.8, 
            'leaf_estimation_iterations': 20,
            'reg_lambda': 0.5,
            'class_weights': class_weights,
            'early_stopping_rounds': 200 
        }
        model = CatBoostClassifier(**params)
        
        model.fit(train_dataset,
                  eval_set=val_dataset, 
                  verbose=100) 
        oof_pred[val_ind] = model.predict_proba(val_dataset) 
        # y_pred += model.predict_proba(df_test[use_features]) / folds.n_splits
        y_pred += model.predict_proba(test_dataset) / folds.n_splits
        
        score = f1_score(y_val, oof_pred[val_ind].argmax(axis=1), average='macro')
        print(f'F1 score: {score}')
        
        print("Features importance...")
        feat_imp = pd.DataFrame({'imp': model.feature_importances_, 'feature': use_features})
        print(feat_imp.sort_values(by='imp').reset_index(drop=True))
        
        del x_train, x_val, y_train, y_val
        gc.collect()
        
    return y_pred, oof_pred

def  macro_f1(target_df: pd.DataFrame,  submit_df: pd.DataFrame)  -> float:

    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """

    weights =  [5/11,  4/11,  1/11,  1/11]

    overall_df = target_df.merge(submit_df, how='left', on=['sn','fault_time'], suffixes=['_gt', '_pr'])
    overall_df.fillna(-1)

    macro_F1 =  0.
    for i in  range(len(weights)):
        TP =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] == i)])
        FP =  len(overall_df[(overall_df['label_gt'] != i) & (overall_df['label_pr'] == i)])
        FN =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
    return macro_F1




