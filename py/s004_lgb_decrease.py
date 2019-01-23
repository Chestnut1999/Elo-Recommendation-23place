#========================================================================
# Args
#========================================================================
import sys
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
    early_stopping_rounds = 100
num_boost_round = 10000
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'purchase_date']

import gc
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
seed_cols = [p for p in params.keys() if p.count('seed')]

params['learning_rate'] = learning_rate
#  params['num_threads'] = 18

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Data Load
base_path = glob.glob('../features/0_base/*.gz')
win_path = '../features/4_winner/*.gz'
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path) + glob.glob('base_path')
tmp_path_list = glob.glob('../features/5_tmp/*.gz')
win_path_list += tmp_path_list

base = utils.read_df_pkl('../input/base*')

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
feature_list = utils.parallel_load_data(path_list=win_path_list)
df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)

#========================================================================
# card_id list by first active month
train_latest_id_list = np.load('../input/card_id_train_first_active_201711.npy')
test_latest_id_list = np.load('../input/card_id_test_first_active_201711.npy')
train = train.loc[train[key].isin(train_latest_id_list), :].reset_index(drop=True)
test = test.loc[test[key].isin(test_latest_id_list), :].reset_index(drop=True)
submit = []
#========================================================================

#========================================================================
# LGBM Setting
model_type='lgb'
metric = 'rmse'
fold=5
seed=1208
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)


train, test, drop_list = LGBM.data_check(train=train, test=test, target=target, encode='dummie', exclude_category=True)

ignore_list = [key, target, 'merchant_id', 'purchase_date']

#========================================================================
# Train & Prediction Start
#========================================================================
import lightgbm as lgb

# TrainとCVのfoldを合わせる為、Train
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

y = train[target]
tmp_train = train.drop(target, axis=1)

#  train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
#  folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
#  kfold = list(folds.split(train,train['outliers'].values))
#  train.drop('outliers', axis=1, inplace=True)

folds = KFold(n_splits=fold, shuffle=True, random_state=seed)
kfold = list(folds.split(train, y))

use_cols = [col for col in train.columns if col not in ignore_list]
valid_feat_list = list(np.random.choice(use_cols, len(use_cols)))
best_valid_list = [100, 100, 100, 100, 100][:int(sys.argv[4])]
best_cv_list = [100, 100, 100, 100, 100][:int(sys.argv[4])]

valid_log_list = []
oof_log = train[[key, target]]
decrease_list = []
all_score_list = []
num_list = []

for i, valid_feat in enumerate([''] + valid_feat_list):

#      logger.info(f'''
#  #========================================================================
#  # Valid{i}/{len(valid_feat_list)} Start!!
#  # Valid Feature: {valid_feat}
#  # Base Valid 1 : {best_valid_list[0]}
#  # Base Valid 2 : {best_valid_list[1]}
#  # Base Valid 3 : {best_valid_list[2]}
#  # Base Valid 4 : {best_valid_list[3]}
#  # Base Valid 5 : {best_valid_list[4]}
#  #========================================================================''')
    update_cnt = 0
    score_list = []
    oof = np.zeros(len(train))

    # One by One Decrease
    if i>0:
        #  valid_cols = list(set(use_cols) - set([valid_feat] + decrease_list))
        valid_cols = list(set(use_cols) - set([valid_feat]))
    else:
        valid_cols = use_cols.copy()

    logger.info(f'''
#========================================================================
# Valid{i}/{len(valid_feat_list)} Start!!
# Valid Feature: {valid_feat} | Feature Num  : {len(valid_cols)} ''')
    for i in range(len(best_cv_list)):
        logger.info(f'''
# Base Valid {i+1} : {best_cv_list[i]} ''')
    logger.info(f'''
#========================================================================''')

    cv_score_list = []
    seed_list = [1208, 605, 328, 1222, 405][:int(sys.argv[4])]
    for seed_num, seed in enumerate(seed_list):

        for seed_p in seed_cols:
            params[seed_p] = seed

        for n_fold, (trn_idx, val_idx) in enumerate(kfold):
            x_train, y_train = tmp_train[valid_cols].loc[trn_idx, :], y.loc[trn_idx]
            x_val, y_val = tmp_train[valid_cols].loc[val_idx, :], y.loc[val_idx]

            x_train.sort_index(axis=1, inplace=True)
            x_val.sort_index(axis=1, inplace=True)

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
            logger.info(f"Validation {n_fold}: RMSE {score}")

        cv_score = np.mean(score_list)
        cv_score_list.append(cv_score)

        if cv_score <  best_cv_list[seed_num]:
            update_cnt+=1
        else:
            break

        logger.info(f"""
# ============================================================
# Seed   : {seed}     | Decrease   : {valid_feat}
# Score  : {cv_score} | Base Score : {best_cv_list[seed_num]}
# Score Update : {cv_score<best_cv_list[seed_num]}
# ============================================================
        """)

    cv_score_avg = np.mean(cv_score_list)
    valid_log_list.append(score_list+[cv_score_avg])
    oof_log[f'valid{i}'] = oof

    if i==0:
        best_cv_list = cv_score_list
        all_score_list.append(cv_score_avg)
        num_list.append(len(all_score_list))
        continue

    # move feature
    if cv_score_avg < np.mean(best_cv_list):
        logger.info(f"""
# ==============================
# Score: {cv_score_avg} | Decrease: {valid_feat} | Score Update!!
# ==============================
        """)
        #  best_cv_list = score_list
        all_score_list.append(cv_score_avg)

        win_path_list = glob.glob(win_path)
        tmp_path_list = glob.glob('../features/5_tmp/*.gz')
        win_path_list += tmp_path_list
        move_list = [path for path in win_path_list if path.count(valid_feat[8:])]
        for move_path in move_list:
            try:
                shutil.move(move_path, '../features/5_tmp/')
            except shutil.Error:
                pass
        decrease_list.append(valid_feat)

    else:
        all_score_list.append(np.nan)

    num_list.append(len(all_score_list))


effect_feat = pd.Series(np.ones(len(valid_feat_list)+1), index=['base'] + valid_feat_list, name='effective')
effect_feat.loc[decrease_list] = 0
effect_feat = effect_feat.to_frame()
effect_feat['score'] = all_score_list
effect_feat['num'] = num_list
effect_feat = effect_feat.drop_duplicates()

df_valid_log = pd.DataFrame(np.array(valid_log_list))
df_valid_log.to_csv(f'../output/{start_time[4:13]}_elo_decrease_valid_log.csv', index=True)

effect_feat.to_csv(f'../output/{start_time[4:13]}_elo_decrease_features.csv', index=True)
