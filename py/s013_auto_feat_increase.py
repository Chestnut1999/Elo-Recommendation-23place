fold=3
fold=5
import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id']
win_path = f'../features/4_winner/*.gz'

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
import shutil
import glob
import re
import gc
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

if model_type=='lgb':
    params = params_elo()[1]
    params['learning_rate'] = learning_rate
params['num_threads'] = 32

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
try:
    sys.argv[5]
    train = train.sample(80000)
except IndexError:
    pass

#========================================================================
#  train_latest_id_list = np.load('../input/card_id_train_first_active_201711.npy')
#  test_latest_id_list = np.load('../input/card_id_test_first_active_201711.npy')
#========================================================================

#========================================================================
# LGBM Setting
seed = 1208
metric = 'rmse'
fold_type='self'
group_col_name=''
dummie=1
oof_flg=True

#========================================================================
# Preprocessing
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
#  train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
train, test, drop_list = LGBM.data_check(train=train, test=[], target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    #  test.drop(drop_list, axis=1, inplace=True)
#========================================================================


#========================================================================
# Increase Valid Features
valid_feat_list = [''] + glob.glob('../features/1_first_valid/*.gz')
#========================================================================

# 全量データで学習する場合
try:
    len(train_latest_id_list)
    pass
except NameError:

    from sklearn.model_selection import StratifiedKFold
    train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
    folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    kfold = list(folds.split(train,train['outliers'].values))
    train.drop('outliers', axis=1, inplace=True)

    #========================================================================
    # outlierに対するスコアを出す
    from sklearn.metrics import mean_squared_error
    out_ids = train.loc[train.target<-30, key].values
    out_val = train.loc[train.target<-30, target].values
    #========================================================================

# Result Input
valid_list = []
used_path = []
df_pred = pd.DataFrame()
result_list = []
add_feat_list = []

#========================================================================
# Experience Start
while len(valid_feat_list)>1:
    train_used_paths = []
    #  test_used_paths = []
    tmp_valid_list = []
    base_cv_score = 100

    for i, path in enumerate(valid_feat_list):

        LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
        LGBM.seed = seed
        cv_score_list = []
        no_update_cnt = 0

        if len(path)>0:
        #  if len(path[0])>0:
            train_path = path
            #  train_path = path[0]
            #  test_path = path[1]
            #  if train_path[-7:] != test_path[-7:]:
                #  print('Feature Sort is different.')
                #  print(train_path, test_path)
                #  sys.exit()
            used_path += list(path).copy()
            train_feat = utils.get_filename(path=train_path, delimiter='gz')
            train_feat = train_feat[14:]
            #  test_feat = utils.get_filename(path=test_path, delimiter='gz')
            #  test_feat = test_feat[14:]

            # 検証するFeatureをデータセットに追加
            try:
                train[train_feat] = utils.read_pkl_gzip(train_path)
                #  test[train_feat] = utils.read_pkl_gzip(test_path)
            except FileNotFoundError:
                continue
            except ValueError:
                continue
        else:
            train_feat = 'base'
            train_path = 'base_path'
            #  test_path = 'base_path'

        # idを絞る
        try:
            tmp_train = train.loc[train[key].isin(train_latest_id_list), :]
            #  tmp_test = test.loc[test[key].isin(test_latest_id_list), :]
            fold_type='kfold'
            kfold = False
            all_id = False
            tmp_train.sort_index(axis=1, inplace=True)
            #  tmp_test.sort_index(axis=1, inplace=True)
        except NameError:
            all_id = True
            train.sort_index(axis=1, inplace=True)
            pass

        logger.info(f'''
#========================================================================
# No: {i}/{len(valid_feat_list)-1}
# Valid Feature: {train_feat}
#========================================================================''')


        #========================================================================
        # Train & Prediction Start
        #========================================================================
        try:
            seed_list = [1208, 605, 328, 1222, 405, 1212, 1012, 1128, 2019][:int(sys.argv[4])]
        except IndexError:
            seed_list = [1208]
        for seed in seed_list:
            LGBM.seed = seed

            if all_id:
                LGBM = LGBM.cross_validation(
                    train=train
                    ,key=key
                    ,target=target
                    ,fold_type=fold_type
                    ,fold=fold
                    ,group_col_name=group_col_name
                    ,params=params
                    ,num_boost_round=num_boost_round
                    ,early_stopping_rounds=early_stopping_rounds
                    ,self_kfold=kfold
                )
            else:
                LGBM = LGBM.cross_validation(
                    train=tmp_train
                    ,key=key
                    ,target=target
                    ,fold_type=fold_type
                    ,fold=fold
                    ,group_col_name=group_col_name
                    ,params=params
                    ,num_boost_round=num_boost_round
                    ,early_stopping_rounds=early_stopping_rounds
                )

            cv_score = LGBM.cv_score
            cv_feim = LGBM.cv_feim
            feature_num = len(LGBM.use_cols)

            cv_score_list.append(cv_score)

            if len(target)>70000:
                if len(df_pred):
                    df_pred['prediction'] += LGBM.prediction
                else:
                    if all_id:
                        df_pred = train.reset_index()[[key, target]].copy()
                    else:
                        df_pred = tmp_train.reset_index()[[key, target]].copy()
                    df_pred['prediction'] = LGBM.prediction

            if cv_score > base_cv_score:
                no_update_cnt += 1

            # 半数以上のCVが更新されなかったら、途中でやめる
            if no_update_cnt > int(len(seed_list)/2):
                break


        if len(path)>0:
            train.drop(train_feat, axis=1, inplace=True)
            #  test.drop(train_feat, axis=1, inplace=True)

            # 半数以上のCVが更新されなかったら、途中でやめる
            if no_update_cnt >= int(len(seed_list)/2):
                continue

        if len(target)>70000:
            df_pred['prediction'] /= len(seed_list)

        cv_score_mean = np.mean(cv_score_list)
        score_update = cv_score_mean < base_cv_score
        logger.info(f'''
#========================================================================
# CV Score Avg : {cv_score_mean} | Base Score: {base_cv_score}
# Score Update : {score_update}
# Valid Feature: {train_feat}
#========================================================================
''')

        # UpdateされなかったFeatureは計算しない
        if not(score_update):
            continue

        #========================================================================
        # Result Summarize

        if len(target)>70000:
            #========================================================================
            # outlierに対するスコアを出す
            out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
            out_score = np.sqrt(mean_squared_error(out_val, out_pred))
            LGBM.val_score_list.append(out_score)
            #========================================================================

        # 結果ファイルの作成
        LGBM.val_score_list.append(cv_score_mean)
        if i:
            feat_name = f"{i}_{train_feat}"
        else:
            feat_name = f"{train_feat}"
        tmp = pd.Series(LGBM.val_score_list, name=feat_name)
        valid_list.append(tmp.copy())
        tmp_valid_list.append(tmp.copy())
        if i==0:
            base_valid = tmp.copy()
            base_cv_score = cv_score_mean

        train_used_paths.append(train_path)
        #  test_used_paths.append(test_path)
        #========================================================================

        # 中間結果の保存
        #  if len(train_used_paths)%10==1 and len(train_used_paths)>9:
        df_valid = pd.concat(valid_list, axis=1)
        print("Enroute Saving...")
        df_valid.to_csv(f'../output/{start_time[4:12]}_elo_multi_feat_valid_lr{learning_rate}.csv', index=True)
        print("Enroute Saving Complete.")
        #  for p in used_path:
        #      shutil.move(p, '../features/2_second_valid/')
        used_path = []
    else:
        #  for p in used_path:
        #      shutil.move(p, '../features/2_second_valid/')
        used_path = []


    #  df_valid = pd.concat(valid_list, axis=1)
    #  df_valid.index = ['cv' if j==fold else f"fold{j}" for j in range(fold+1)]
    #  for col in df_valid.columns:
    #      if col.count('base'):continue
    #      df_valid[f"val_{col}"] = (df_valid[col].values < base_valid.values) * 1
    #  df_valid.to_csv(f'../output/{start_time[4:12]}_elo_multi_feat_valid_lr{learning_rate}.csv', index=False)
    #  del df_valid
    #  gc.collect()

    # 今ループの結果検証
    df_valid = pd.concat(tmp_valid_list, axis=1)
    base_score = df_valid['base'].iloc[len(df_valid)-1]
    df_cv = df_valid.T
    best_feature = df_cv.iloc[:, len(df_valid)-1].idxmin()
    df_cv = df_cv.iloc[:, len(df_valid)-1].to_frame().reset_index()
    df_cv.columns = ['feature', 'score']

    logger.info(f'''
#========================================================================
# Feature Selection Loop End.
# Base Score: {base_score}
# Best Score: {df_cv['score'].min()}
# Best Feat : {best_feature}
#========================================================================''')
    # Base Scoreを更新したFeatureに絞る
    candidates = df_cv[df_cv['score']<base_score]

    logger.info(f'''
#========================================================================
# Candidate Shape: {candidates.shape}
#========================================================================''')

    # 検証するFeature Listの初期化
    valid_feat_list = ['']
    #  test_feat_list = ['']

    # Base Scoreを更新したFeatureがなければBreak
    if len(candidates)==0:
        break
    elif str(type(candidates)).count('Series'):
        candidates = candidates.to_frame().T
    else:
        candidates.sort_values(by='score', ascending=True, inplace=True)

    # candidatesからTOPのfeature名とPATHを取り出すには、idxmaxかSortしたTOPのindexをとる。今回はSort
    update_idx_list = list(candidates.index)
    add_feat_idx = update_idx_list[0]
    #  move_path = [train_used_paths[add_feat_idx], test_used_paths[add_feat_idx]]
    move_path = train_used_paths[add_feat_idx]

    # 移動させるFeatureのパスを特定する為に必要
    add_feat_name = re.search(r'/([^/.]*).gz', move_path).group(1)

    # 最後の結果ログ用
    add_feat_list.append(add_feat_name)

    # ベストスコアを更新したFeatureのみ移動させる
    feat_list = glob.glob('../features/1_first_valid/*.gz')
    for feat in feat_list:
        filename = re.search(r'/([^/.]*).gz', feat).group(1)
        if filename.count(add_feat_name[:7]) and filename.count(add_feat_name[14:]):
            #  shutil.move(move_path, '../features/4_winner/')
            if feat.count('train'):
                train[add_feat_name[:7] + '_' + add_feat_name[14:]] = utils.read_pkl_gzip(feat)
            shutil.move(feat, '../features/5_tmp/')

            logger.info(f"""
#========================================================================
# Move Feature: {candidates['feature'].values[0]}
# PATH        : {feat}
#======================================================================== """)

    # ベストスコアの更新とはならなかったFeatureは、ベストFeatureをデータセットに追加した
    # 後に再度検証するので、リスト追加する
    if len(update_idx_list)>1:
        for add_feat_idx in update_idx_list[1:]:
            valid_feat_list.append(train_used_paths[add_feat_idx])
            #  test_feat_list.append(test_used_paths[add_feat_idx])

    # 最後の結果ファイル可視化ようの情報を追加する
    candidates['add_no'] = len(add_feat_list)+1
    add_flg_list = [1] + list(np.zeros(len(candidates)-1).astype('int8'))
    candidates['add_no'] = add_flg_list
    result_list.append(candidates)

    logger.info(f"""
#========================================================================
# Remain Valid Feature: {len(valid_feat_list)-1}
# This Time Top5 :
{candidates.head()}
#======================================================================== """)
    # 途中結果の保存
    if len(result_list):
        result = pd.concat(result_list, axis=0).reset_index(drop=True)
        result.to_csv(f'../output/{start_time[4:12]}_elo_multi_feat_valid_lr{learning_rate}.csv', index=False)

    # 追加検証するFeatureがなければ終了
    if len(valid_feat_list)==1:
        break

# 結果の保存
if len(result_list):
    result = pd.concat(result_list, axis=0).reset_index(drop=True)
    result.to_csv(f'../output/{start_time[4:12]}_elo_multi_feat_valid_lr{learning_rate}.csv', index=False)

logger.info(f'''
#========================================================================
# Increase Features:
{pd.Series(add_feat_list)}
#========================================================================''')
