#========================================================================
# Args
#========================================================================
learning_rate = 0.01
learning_rate = 0.5
early_stopping_rounds = 200
early_stopping_rounds = 2
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

params = params_elo()[1]
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
train_feature_list = utils.parallel_load_data(path_list=train_path_list)
test_feature_list = utils.parallel_load_data(path_list=test_path_list)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)
test = pd.concat(test_feature_list, axis=1)
test = pd.concat([base_test, test], axis=1)
#========================================================================

#========================================================================
# LGBM Setting
model_type='lgb'
metric = 'rmse'
fold=3
seed=1208
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)


train, test, drop_list = LGBM.data_check(train=train, test=test, target=target, encode='dummie', exclude_category=True)

ignore_list = [key, target, 'merchant_id', 'purchase_date']

#========================================================================
# Train & Prediction Start
#========================================================================
import lightgbm as lgb

# TrainとCVのfoldを合わせる為、Train
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

y = train[target]
tmp_train = train.drop(target, axis=1)

train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
kfold = list(folds.split(train,train['outliers'].values))
train.drop('outliers', axis=1, inplace=True)

use_cols = [col for col in train.columns if col not in ignore_list]
valid_feat_list = list(np.random.choice(use_cols, len(use_cols)))
best_valid_list = [100, 100, 100]

valid_log_list = []
oof_log = train[[key, target]]
decrease_list = []
all_score_list = []

for i, valid_feat in enumerate([''] + valid_feat_list):

    logger.info(f'''
#========================================================================
# Valid{i}/{len(valid_feat_list)} Start!!
# Valid Feature: {valid_feat}
# Base Valid 1 : {best_valid_list[0]}
# Base Valid 2 : {best_valid_list[1]}
# Base Valid 3 : {best_valid_list[2]}
#========================================================================''')
    update_cnt = 0
    score_list = []
    oof = np.zeros(len(train))

    # One by One Decrease
    if i>0:
        valid_cols = list(set(use_cols) - set([valid_feat] + decrease_list))
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
    all_score_list.append(np.mean(score_list))

    if i==0:
        best_valid_list = score_list
        #  feim = pd.Series(lgbm.feature_importance(), name='importance', index=valid_cols)
        #  feim.sort_values(inplace=True)
        continue

    # move feature
    if update_cnt>=2:
        logger.info(f"""
# ==============================
# Score Update!!
# Decrease :{valid_feat}
# Score    : {np.mean(score_list)}
# ==============================
        """)
        #  best_valid_list = score_list
        path_list = glob.glob(win_path)
        move_list = [path for path in path_list if path.count(valid_feat[8:])]
        for move_path in move_list:
            shutil.move(move_path, '../features/5_tmp/')
        decrease_list.append(valid_feat)

effect_feat = pd.Series(np.ones(len(valid_feat_list)+1), index=['base'] + valid_feat_list, name='effective')
effect_feat.loc[decrease_list] = 0
effect_feat = effect_feat.to_frame()
effect_feat['score'] = all_score_list

df_valid_log = pd.DataFrame(np.array(valid_log_list))
df_valid_log.to_csv(f'../output/{start_time[4:11]}_decrease_valid_log.csv', index=True)

effect_feat.to_csv(f'../output/{start_time[4:11]}_decrease_features.csv', index=True)
