fold=5
#  params['num_threads'] = 17
import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id']

win_path = f'../features/4_winner/*.gz'
stack_name='en_route'
fname=''
xray=False
#  xray=True
submit = pd.read_csv('../input/sample_submission.csv')
#  submit = []

#========================================================================
# argv[1] : model_type 
# argv[2] : learning_rate
# argv[3] : early_stopping_rounds
#========================================================================

try:
    model_type=sys.argv[1]
except IndexError:
    model_type='lgb'
try:
    learning_rate = float(sys.argv[2])
except IndexError:
    learning_rate = 0.1
try:
    early_stopping_rounds = int(sys.argv[3])
except IndexError:
    early_stopping_rounds = 150
num_boost_round = 10000

import numpy as np
import datetime
import glob
import gc
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

if model_type=='lgb':
    params = params_elo()[1]
    params['learning_rate'] = learning_rate

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path)
train_path_list = []
test_path_list = []
for path in win_path_list:
    if path.count('train'):
        train_path_list.append(path)
    elif path.count('test'):
        test_path_list.append(path)

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
train_feature_list = utils.parallel_load_data(path_list=train_path_list)
test_feature_list = utils.parallel_load_data(path_list=test_path_list)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)
test = pd.concat(test_feature_list, axis=1)
test = pd.concat([base_test, test], axis=1)

# Valid Features
feat_list = glob.glob('../features/1_first_valid/*.gz')
train_feat_list = ['']
test_feat_list = ['']

for path in feat_list:
    if path.count('train'):
        train_feat_list.append(path)
    elif path.count('test'):
        test_feat_list.append(path)

#========================================================================

#========================================================================
# LGBM Setting
seed = 1208

train[target] = train[target].map(lambda x: 1 if x<-30 else 0)
metric = 'auc'
params['objective'] = 'binary'
params['metric'] = metric
fold_type='kfold'
group_col_name=''
dummie=1
oof_flg=True
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)

#========================================================================

valid_list = []
for i, path in enumerate(zip(train_feat_list, test_feat_list)):

    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    LGBM.seed = seed
    #========================================================================

    if len(path[0])>0:
        train_path = path[0]
        test_path = path[1]
        train_feat = utils.get_filename(path=train_path, delimiter='gz')
        train_feat = train_feat[14:]
        test_feat = utils.get_filename(path=test_path, delimiter='gz')
        test_feat = test_feat[13:]

        train[train_feat] = utils.read_pkl_gzip(train_path)
        test[train_feat] = utils.read_pkl_gzip(test_path)
    else:
        train_feat = 'base'

    logger.info(f'''
    #========================================================================
    # No: {i}/{len(train_feat_list)}
    # Valid Feature: {train_feat}
    #========================================================================''')

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

    cv_score = LGBM.cv_score
    cv_feim = LGBM.cv_feim
    feature_num = len(LGBM.use_cols)
    df_pred = LGBM.result_stack.copy()

    if len(path[0])>0:
        train.drop(train_feat, axis=1, inplace=True)
        test.drop(train_feat, axis=1, inplace=True)

    LGBM.val_score_list.append(cv_score)
    tmp = pd.Series(LGBM.val_score_list, name=f"{i}_{train_feat}")
    valid_list.append(tmp.copy())
    if i==0:
        base_valid = tmp.copy()

    if i%10==1 and i>9:
        df_valid = pd.concat(valid_list, axis=1)
        print("Enroute Saving...")
        df_valid.to_csv(f'../output/{start_time[4:11]}_elo_clf_multi_feat_valid_lr{learning_rate}.csv', index=True)
        print("Enroute Saving Complete.")

df_valid = pd.concat(valid_list, axis=1)

for col in df_valid.columns:
    if col.count('base'):continue
    df_valid[f"val_{col}"] = (df_valid[col].values > base_valid.values) * 1

df_valid.to_csv(f'../output/{start_time[4:11]}_elo_clf_multi_feat_valid_lr{learning_rate}.csv', index=True)
