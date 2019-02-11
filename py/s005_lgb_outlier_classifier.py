import gc
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'hist_regist_term']

win_path = f'../features/4_winner/*.gz'
stack_name='outlier_classify'
fname=''

#========================================================================
# argv[1] : model_type 
# argv[2] : learning_rate
# argv[3] : early_stopping_rounds
#========================================================================

try:
    learning_rate = float(sys.argv[1])
except IndexError:
    learning_rate = 0.01
early_stopping_rounds = 200
num_boost_round = 10000
num_threads = 36

import numpy as np
import datetime
import glob
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from params_lgbm import params_elo
from pdp import Xray_Cal
sys.path.append(f'{HOME}/kaggle/data_analysis')
from model.lightgbm_ex import lightgbm_ex as lgb_ex

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

model_type = 'lgb'
params = params_elo()[1]
params['learning_rate'] = learning_rate
# Best outlier fit LB3.690
#  num_leaves = 4
#  num_leaves = 16
num_leaves = 31
num_leaves = 48
num_leaves = 59
num_leaves = 61
num_leaves = 68
num_leaves = 70
params['num_leaves'] = num_leaves
params['num_threads'] = num_threads
if num_leaves>=70:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.325582
    params['min_child_samples'] = 30
    params['lambda_l2'] = 7
elif num_leaves>65:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2755158
    params['min_child_samples'] = 37
    params['lambda_l2'] = 7

elif num_leaves>60:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2792
    params['min_child_samples'] = 59
    params['lambda_l2'] = 2

elif num_leaves>50:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.256142
    params['min_child_samples'] = 55
    params['lambda_l2'] = 3

elif num_leaves>40:
    params['subsample'] = 0.8757099996397999
    #  params['colsample_bytree'] = 0.7401342964627846
    params['colsample_bytree'] = 0.3
    params['min_child_samples'] = 50

else:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.3
    params['min_child_samples'] = 30


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load

win_path = f'../features/4_winner/*.gz'
win_path = f'../model/LB3670_70leaves_colsam0322/*.gz'
#  tmp_path_list = glob.glob(f'../features/5_tmp/*.gz') + glob.glob(f'../features/0_exp/*.gz')
tmp_path_list = glob.glob(f'../features/5_tmp/*.gz')
win_path_list = glob.glob(win_path) + tmp_path_list
win_path_list = glob.glob(win_path)

col_term = 'hist_regist_term'
base = utils.read_df_pkl('../input/base_term*')[[key, 'first_active_month', target, col_term]]
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

feature_list = utils.parallel_load_data(path_list=win_path_list)

df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
#========================================================================

#========================================================================
# LGBM Setting
try:
    seed_list = np.arange(int(sys.argv[2]))
    seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005]
except IndexError:
    seed_list = [1208]


train[target] = train[target].map(lambda x: 1 if x<-30 else 0)
#  train[target] = train[target].map(lambda x: 1 if x>1.5 else 0)
#  train = pd.read_csv('../features/loy_0_1.csv')
#  train[target] = train[target].map(lambda x: 1 if x<1 else 0)


metric = 'auc'
metric = 'binary_logloss'
params['objective'] = 'binary'
params['metric'] = metric
fold=6
#  fold_type='stratified'
fold_type='kfold'
#  fold_type='self'
group_col_name=''
dummie=1
oof_flg=True
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
#  if len(drop_list):
#      train.drop(drop_list, axis=1, inplace=True)
#      test.drop(drop_list, axis=1, inplace=True)

train[col_term] = train[col_term].map(lambda x: 
                                          6 if 6<=x and x<=8  else 
                                          9 if 9<=x and x<=12
                                          else x
                                         )
#========================================================================
# seed_avg
seed_pred = np.zeros(len(test))
cv_list = []
for i, seed in enumerate(seed_list):

    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    LGBM.seed = seed

    if key not in train.columns:
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

#========================================================================
# Train & Prediction Start
#========================================================================
    LGBM = LGBM.cross_prediction(
        train=train
        ,test=test
        ,key=key
        ,target=target
        ,fold_type=fold_type
        ,fold=fold
        ,group_col_name=group_col_name
        ,params=params
        ,num_boost_round=num_boost_round
        ,early_stopping_rounds=early_stopping_rounds
        ,oof_flg=oof_flg
    )

    seed_pred += LGBM.prediction

    if i==0:
        cv_list.append(LGBM.cv_score)
        cv_feim = LGBM.cv_feim
        feature_num = len(LGBM.use_cols)
        df_pred = LGBM.result_stack.copy()
    else:
        cv_score = LGBM.cv_score
        cv_list.append(cv_score)
        LGBM.cv_feim.columns = [col if col.count('feature') else f"{col}_{seed}" for col in LGBM.cv_feim.columns]
        cv_feim = cv_feim.merge(LGBM.cv_feim, how='inner', on='feature')
        df_pred = df_pred.merge(LGBM.result_stack.rename(columns={'prediction':f'prediction_{i}'}), how='inner', on=key)

#========================================================================
# Result
#========================================================================
test_pred = seed_pred / len(seed_list)
cv_score = np.mean(cv_list)

cv_feim.to_csv(f'../valid/{start_time[4:12]}_{model_type}_{fname}_feat{feature_num}_binary_CV{cv_score}_lr{learning_rate}.csv', index=False)

#========================================================================
# STACKING

if len(stack_name)>0:
    logger.info(f'result_stack shape: {df_pred.shape}')
    if len(seed_list)==1:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_binary_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)
    else:
        pred_cols = [col for col in df_pred.columns if col.count('predict')]
        df_pred['pred_mean'] = df_pred[pred_cols].mean(axis=1)
        df_pred['pred_std'] = df_pred[pred_cols].std(axis=1)
        drop_cols = [col for col in df_pred.columns if col.count('target_')]
        if len(drop_cols)>0:
            df_pred.drop(drop_cols, axis=1, inplace=True)
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{len(seed_list)}seed_{model_type}_binary_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)

#========================================================================
