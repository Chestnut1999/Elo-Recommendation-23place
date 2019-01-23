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
params['learning_rate'] = learning_rate
params['num_threads'] = 32
seed_cols = [p for p in params.keys() if p.count('seed')]


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Data Load
win_path = '../features/4_winner/*.gz'
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path)
# tmp_path_listには検証中のfeatureを入れてある
tmp_path_list = glob.glob('../features/5_tmp/*.gz')
win_path_list += tmp_path_list

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
feature_list = utils.parallel_load_data(path_list=win_path_list)
df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
#  test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
test = []
#  try:
#      sys.argv[5]
#      train = train.sample(80000).reset_index(drop=True)
#  except IndexError:
#      pass

#========================================================================

#========================================================================
# card_id list by first active month
#  train_latest_id_list = np.load('../input/card_id_train_first_active_201712.npy')
#  test_latest_id_list = np.load('../input/card_id_test_first_active_201712.npy')
#  train = train.loc[train[key].isin(train_latest_id_list), :].reset_index(drop=True)
#  test = test.loc[test[key].isin(test_latest_id_list), :].reset_index(drop=True)
#========================================================================

#========================================================================
# LGBM Setting
model_type='lgb'
metric = 'rmse'
fold=5
seed=1208
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)


train, test, drop_list = LGBM.data_check(train=train, test=[], target=target, encode='dummie', exclude_category=True)

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

try:
    len(train_latest_id_list)
    folds = KFold(n_splits=fold, shuffle=True, random_state=seed)
    kfold = list(folds.split(train, y))
except NameError:
    train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
    folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    kfold = list(folds.split(train,train['outliers'].values))
    train.drop('outliers', axis=1, inplace=True)


use_cols = [col for col in train.columns if col not in ignore_list]
first_num = len(use_cols)
valid_feat_list = list(np.random.choice(use_cols, len(use_cols)))
remove_list = []
result_list = []

while len(valid_feat_list):

    # Scoreを初期化
    best_valid_list = [100, 100, 100, 100, 100, 100, 100, 100, 100][:int(sys.argv[4])]
    best_cv_list = [100, 100, 100, 100, 100, 100, 100, 100, 100][:int(sys.argv[4])]
    seed_list = [1208, 605, 328, 1222, 405, 1212, 1012, 1128, 2019][:int(sys.argv[4])]

    valid_log_list = []
    oof_log = train[[key, target]]
    decrease_list = []
    all_score_list = []
    num_list = []

    for i, valid_feat in enumerate([''] + valid_feat_list):

        update_cnt = 0
        score_list = []
        oof = np.zeros(len(train))

        # One by One Decrease
        if i>0:
            valid_cols = list(set(use_cols) - set([valid_feat]))
        else:
            valid_cols = use_cols.copy()

        logger.info(f'''
#========================================================================
# Valid{i}/{len(valid_feat_list)} Start!!
# Valid Feature: {valid_feat} | Num Current/First: {len(valid_cols)}/{first_num}
''')
        for cv_num, base_cv in enumerate(best_cv_list):
            logger.info(f'Base CV{cv_num}: {base_cv}')
        logger.info(f'''
#========================================================================''')

        cv_score_list = []
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
# Feature Num : {i}/{len(valid_cols)} | First Num: {first_num}
# Decrease    : {valid_feat} | Seed: {seed}
# Score       : {cv_score} | Base Score: {best_cv_list[seed_num]}
# Score Update: {cv_score<best_cv_list[seed_num]}
# ============================================================
            """)

        cv_score_avg = np.mean(cv_score_list)
        valid_log_list.append(score_list+[cv_score_avg])
        oof_log[f'valid{i}'] = oof


        if i==0:
            best_cv_list = np.array(best_cv_list)
            cv_score_list = np.array(cv_score_list)
            # 半分以上のCVを更新しなかったら、CVリストを据え置き
            if ( (best_cv_list > cv_score_list).sum()  < len(seed_list)/2 ) and cv_score_avg < np.mean(best_cv_list) :
                pass
            else:
                best_cv_list = cv_score_list
            all_score_list.append(cv_score_avg)

        elif cv_score_avg < np.mean(best_cv_list):
            logger.info(f"""
# ============================================================
# Score Update!!
# Decrease : {valid_feat}
# Score    : {cv_score_avg} | Base: {np.mean(best_cv_list)}
# ============================================================
            """)
            #  best_cv_list = score_list
            all_score_list.append(cv_score_avg)

            # 候補となったfeatureを移動させる
            #  win_path_list = glob.glob(win_path)
            #  tmp_path_list = glob.glob('../features/5_tmp/*.gz')
            #  win_path_list += tmp_path_list
            #  move_list = [path for path in win_path_list if path.count(valid_feat[8:])]
            #  for move_path in move_list:
            #      try:
            #          shutil.move(move_path, '../features/5_tmp/')
            #      except shutil.Error:
            #          pass
            decrease_list.append(valid_feat)

        else:
            all_score_list.append(np.nan)
            logger.info(f"""
# ============================================================
# Not Score Update...
# Decrease : {valid_feat}
# Score    : {cv_score_avg} | Base: {np.mean(best_cv_list)}
# ============================================================
            """)

        num_list.append(len(all_score_list))

    # 全featureの検証が終わったら入る
    effect_feat = pd.Series(np.ones(len(valid_feat_list)+1), index=['base'] + valid_feat_list, name='flg')
    effect_feat.loc[decrease_list] = 0
    effect_feat = effect_feat.to_frame().reset_index()
    effect_feat.columns = ['feature', 'flg']
    effect_feat['score'] = all_score_list
    effect_feat['num'] = num_list
    effect_feat = effect_feat.drop_duplicates()
    base_score = effect_feat[effect_feat['feature']=='base']['score'].values[0]
    candidates = effect_feat[effect_feat['score'] < base_score]

    # Base Scoreを超えていたら、Selectionを継続
    if len(candidates)==0:
        valid_feat_list = []
    else:
        candidates.sort_values(by='score', ascending=True, inplace=True)
        rm_flg_list = np.zeros(len(candidates))
        rm_flg_list[0] = 1
        valid_feat_list = candidates['feature'].values

        candidates['remove_flg'] = rm_flg_list
        candidates['remove_no'] = len(result_list)
        effect_feat['remove_no'] = len(result_list)
        #  result_list.append(candidates)
        result_list.append(effect_feat)

        # ベストスコアを更新したfeatureをremove_listに追加する
        remove_list.append(valid_feat_list[0])
        logger.info(f'''
#========================================================================
# Enroute Remove Features: {valid_feat_list[0]}
#========================================================================''')

        # 先頭のfeatureをremove_listに追加したfeatureは除外
        valid_feat_list = list(valid_feat_list[1:])
        use_cols = [col for col in train.columns if col not in ignore_list+remove_list]

        # 途中経過保存
        if len(result_list):
            result = pd.concat(result_list, axis=0)
            result.to_csv(f'../output/{start_time[4:13]}_elo_decrease_features_lr{learning_rate}.csv')

logger.info(f'''
#========================================================================
# Final Remove Features:
{pd.Series(remove_list)}
#========================================================================
''')

if len(result_list):
    result = pd.concat(result_list, axis=0)
    result.to_csv(f'../output/{start_time[4:13]}_elo_decrease_features_lr{learning_rate}.csv')


win_path_list = glob.glob(win_path)
tmp_path_list = glob.glob('../features/5_tmp/*.gz')
win_path_list += tmp_path_list
for path in win_path_list:
    for feature in remove_list:
        if path.count(feature[8:]):
            try:
                shutil.move(path, '../features/3_third_valid/')
                #  shutil.move(path, '../features/1_first_valid/')
            except shutil.Error:
                pass
