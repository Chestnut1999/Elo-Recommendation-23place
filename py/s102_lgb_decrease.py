#========================================================================
# Args
#========================================================================
learning_rate = 0.1
early_stopping_rounds = 100
num_boost_round = 10000
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'purchase_date']

import gc
import sys
import numpy as np
import pandas as pd
import datetime

import shutil
import glob
import os
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

params = params_elo()
params['learning_rate'] = learning_rate

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Data Load
win_path = '../features/4_winner/*.gz'
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
train_feature_list = utils.pararell_load_data(path_list=train_path_list)
test_feature_list = utils.pararell_load_data(path_list=test_path_list)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)
test = pd.concat(test_feature_list, axis=1)
test = pd.concat([base_test, test], axis=1)
#========================================================================

#========================================================================
# LGBM Setting
model_type='lgb'
metric = 'rmse'
fold=2
seed=1208
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)


train, test, drop_list = LGBM.data_check(train=train, test=test, target=target, encode='dummie', exclude_category=True)

ignore_list = [key, target, 'merchant_id', 'purchase_date']

#========================================================================
# Train & Prediction Start
#========================================================================
import lightgbm as lgb

# TrainとCVのfoldを合わせる為、Train
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

y = train[target]
tmp_train = train.drop(target, axis=1)

folds = KFold(n_splits=fold, shuffle=True, random_state=seed)
kfold = list(folds.split(tmp_train, y))

use_cols = [col for col in train.columns if col not in ignore_list]
valid_feat_list = use_cols.copy()
best_valid_list = [3.795892051152665, 3.768752515103204]

valid_log_list = []
oof_log = train[[key, target]]
decrease_list = []

for i, valid_feat in enumerate([''] + valid_feat_list):

    logger.info(f'''
#========================================================================
# Valid{i}/{len(valid_feat_list)} Start!!
# Valid Feature: {valid_feat}
# Best Valid 1 : {best_valid_list[0]}
# Best Valid 2 : {best_valid_list[1]}
#========================================================================''')
    update_cnt = 0
    score_list = []
    oof = np.zeros(len(train))

    # One by One Decrease
    if len(valid_feat)>0:
        valid_cols = list(set(use_cols) - set([valid_feat]))
    else:
        valid_cols = use_cols.copy()

    for n_fold, (trn_idx, val_idx) in enumerate(kfold):
        x_train, y_train = tmp_train[valid_cols].loc[trn_idx, :], y.loc[trn_idx]
        x_val, y_val = tmp_train[valid_cols].loc[val_idx, :], y.loc[val_idx]

        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_eval = lgb.Dataset(data=x_val, label=y_val)

        lgbm = lgb.train(
            train_set=lgb_train,
            valid_sets=lgb_eval,
            params=params,
            verbose_eval=200,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
        )

        y_pred = lgbm.predict(x_val)
        oof[val_idx] = y_pred

        score = np.sqrt(mean_squared_error(y_val, y_pred))
        score_list.append(score)
        if score <  best_valid_list[n_fold]:
            update_cnt+=1
        else:
            break
        logger.info(f"Validation {n_fold}: RMSE {score}")

    valid_log_list.append(score_list+[np.mean(score_list)])
    oof_log[f'valid{i}'] = oof

    if len(valid_feat)==0:
        best_valid_list = score_list
        continue

    # move feature
    if update_cnt==fold:
        logger.info(f"""
# ==============================
# Score Update!!
# Decrease :{valid_feat}
# Score    : {np.mean(score_list)}
# ==============================
        """)
        best_valid_list = score_list
        path_list = glob.glob(win_path)
        move_list = [path for path in path_list if path.count(valid_feat)]
        for move_path in move_list:
            shutil.move(move_path, '../features/9_gdrive/')
        decrease_list.append(valid_feat)

effect_feat = pd.Series(np.ones(len(use_cols)), index=use_cols, name='effective')
effect_feat.loc[decrease_list] = 0

df_valid_log = pd.DataFrame(np.array(valid_log_list))
df_valid_log.to_csv('../output/{start_time[4:9]}_decrease_valid_log.csv', index=True)
