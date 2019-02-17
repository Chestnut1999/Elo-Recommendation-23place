fold_seed = 328
outlier_thres = -3
num_threads = 32
#  num_threads = 36
import sys
import pandas as pd
try:
    model_no = int(sys.argv[1])
except IndexError:
    model_no = 0
valid_type = sys.argv[2]

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
col_term = 'hist_regist_term'
no_flg = 'no_out_flg'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term', col_term, no_flg, 'clf_pred']
stack_name='stack'

model_type='lgb'
learning_rate = 0.01
early_stopping_rounds = 200
num_boost_round = 5000

import numpy as np
import datetime
import glob
import re
import gc
import os
from sklearn.metrics import mean_squared_error
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from params_lgbm import params_elo
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

params = params_elo()[1]
params['learning_rate'] = learning_rate
params['objective'] = 'binary'

# Best outlier fit LB3.690
#  num_leaves = 4
#  num_leaves = 16
num_leaves = 31
num_leaves = 48
num_leaves = 57
#  num_leaves = 59
#  num_leaves = 61
#  num_leaves = 68
#  num_leaves = 70
#  num_leaves = 71
try:
    num_leaves = sys.argv[4]
except IndexError:
    num_leaves = 57

params['num_leaves'] = num_leaves
params['num_threads'] = num_threads
if num_leaves==71:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.3180226
    params['min_child_samples'] = 31
    params['lambda_l2'] = 14
elif num_leaves==70:
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

elif num_leaves==57:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2513
    params['min_child_samples'] = 32
    params['lambda_l2'] = 9

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

colsample_bytree = params['colsample_bytree']
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Data Load

win_path = f'../features/4_winner/*.gz'
model_path_list = [f'../model/LB3670_70leaves_colsam0322/*.gz', '../model/E2_lift_set/*.gz', '../model/E3_PCA_set/*.gz', '../model/E4_mix_set/*.gz']

model_path = model_path_list[model_no]
win_path_list = glob.glob(model_path)

base = utils.read_pkl_gzip('../input/base_no_out_clf.gz')[[key, target, col_term, 'first_active_month', no_flg, 'clf_pred']]
#  base = utils.read_df_pkl('../input/base_term*')[[key, target, col_term, 'first_active_month']]
base[col_term] = base[col_term].map(lambda x:
                                          24 if 19<=x else
                                          18 if 16<=x and x<=18 else
                                          15 if 13<=x and x<=15 else
                                          12 if 9<=x and x<=12  else
                                          8 if 6<=x and x<=8    else
                                          5 if x==5 else
                                          4
                                         )
#  nn_stack_plus = utils.read_pkl_gzip('../ensemble/NN_ensemble/0213_142_elo_NN_stack_E1_row99239_outpart-all_235feat_const1_lr0.001_batch128_epoch30_CV1.2724309982670599.gz')[[key, 'prediction']].set_index(key)
#  nn_stack_minus = utils.read_pkl_gzip('../ensemble/NN_ensemble/0213_145_elo_NN_stack_E1_row104308_outpart-all_235feat_const1_lr0.001_batch128_epoch30_CV4.864183650939903.gz')[[key, 'prediction']].set_index(key)
#  base.set_index(key, inplace=True)
#  base['nn_plus'] = nn_stack_plus['prediction']
#  base['nn_minus'] = nn_stack_minus['prediction']
#  base.reset_index(inplace=True)

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

feature_list = utils.parallel_load_data(path_list=win_path_list)

df_feat = pd.concat(feature_list, axis=1)

train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
train[target] = train[target].map(lambda x: 1 if x<-30 else 0)
y = train[target].values
#  train[col_term] = train[col_term].map(lambda x: 
#                                            6 if 6<=x and x<=8  else 
#                                            9 if 9<=x and x<=12
#                                            else x
#                                           )
#========================================================================

#========================================================================
# LGBM Setting
try:
    argv3 = int(sys.argv[3])
    seed_list = np.arange(argv3)
    if argv3<=10:
        seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005, 2019][:argv3]
        seed_list = [328, 605, 1212, 1222, 405, 1128, 1012, 1208, 2005, 2019][:argv3]
except IndexError:
    seed_list = [1208]
    seed_list = [328]

metric = 'auc'
metric = 'binary_logloss'
params['objective'] = 'binary'
params['metric'] = metric
group_col_name=''
dummie=1
oof_flg=True
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)


fold=6
kfold_path = f'../input/kfold_{valid_type}_all_fold{fold}_seed{fold_seed}.gz'
if os.path.exists(kfold_path):
    kfold = utils.read_pkl_gzip(kfold_path)
    fold_type='self'
else:
    fold_type='stratified'
    kfold = False

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
        ,self_kfold = kfold
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
cv_feim.to_csv(f'../valid/{start_time[4:12]}_{model_type}_{stack_name}_feat{feature_num}_binary_CV{cv_score}_lr{learning_rate}.csv', index=False)

#========================================================================
# STACKING

if len(stack_name)>0:
    logger.info(f'result_stack shape: {df_pred.shape}')
    if len(seed_list)==1:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_binary_valid-{valid_type}_seed{len(seed_list)}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred.reset_index()[[key, 'prediction']])
    else:
        pred_cols = [col for col in df_pred.columns if col.count('predict')]
        df_pred['pred_mean'] = df_pred[pred_cols].mean(axis=1)
        df_pred['pred_std'] = df_pred[pred_cols].std(axis=1)
        drop_cols = [col for col in df_pred.columns if col.count('target_')]
        if len(drop_cols)>0:
            df_pred.drop(drop_cols, axis=1, inplace=True)
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_binary_valid-{valid_type}_seed{len(seed_list)}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred.reset_index()[[key, 'pred_mean']])

#========================================================================
