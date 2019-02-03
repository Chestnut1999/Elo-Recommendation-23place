fold=3
fold=5
num_threads = -1
import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index']
win_path = f'../features/4_winner/*.gz'

try:
    learning_rate = float(sys.argv[1])
except IndexError:
    learning_rate = 0.1
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

model_type='lgb'
params = params_elo()[1]

num_leaves = 31
num_leaves = 48
#  num_leaves = 63
params['num_leaves'] = num_leaves
params['num_threads'] = num_threads
params['learning_rate'] = learning_rate
if num_leaves>40:
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
win_path = '../features/4_winner/*.gz'
base = utils.read_df_pkl('../input/base_first*0*')
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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
# 全量データで学習する場合
if sys.argv[3]=='ods':

    #========================================================================
    # ods.ai 3rd kernel
    # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78903
    # KFold: n_splits=6(or 7)!, shuffle=False!
    #========================================================================
    train['rounded_target'] = train['target'].round(0)
    train = train.sort_values('rounded_target').reset_index(drop=True)
    vc = train['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df = pd.DataFrame()
    train['indexcol'],idx = 0,1
    for k,v in vc.items():
        step = train.shape[0]/v
        indent = train.shape[0]/(v+1)
        df2 = train[train['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j*step + 0.000001*idx
        df = pd.concat([df2,df])
        idx+=1
    train = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del train['indexcol'], train['rounded_target']
    fold_type = 'self'
    fold = 6
    folds = KFold(n_splits=fold, shuffle=False, random_state=seed)
    kfold = list(folds.split(train, train[target].values))


#========================================================================
# outlierに対するスコアを出す
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
            used_path += list(path).copy()
            valid_feat = utils.get_filename(path=path, delimiter='gz')

            # 検証するFeatureをデータセットに追加
            try:
                train[valid_feat] = utils.read_pkl_gzip(path)[:len(base_train)]
            except FileNotFoundError:
                continue
            except ValueError:
                continue
        else:
            valid_feat = 'base'
            path = 'base_path'

        # idを絞る
        train.sort_index(axis=1, inplace=True)

        logger.info(f'''
#========================================================================
# No: {i}/{len(valid_feat_list)-1}
# Valid Feature: {valid_feat}
#========================================================================''')

        #========================================================================
        # Train & Prediction Start
        #========================================================================
        try:
            seed_list = [1208, 605, 328, 1222, 405, 1212, 1012, 1128, 2019][:int(sys.argv[2])]
        except IndexError:
            seed_list = [1208]
        for seed in seed_list:
            LGBM.seed = seed

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

            cv_score = LGBM.cv_score
            cv_feim = LGBM.cv_feim
            feature_num = len(LGBM.use_cols)

            cv_score_list.append(cv_score)

            if len(df_pred):
                df_pred['prediction'] += LGBM.prediction
            else:
                df_pred = train.reset_index()[[key, target]].copy()
                df_pred['prediction'] = LGBM.prediction

            if cv_score > base_cv_score:
                no_update_cnt += 1

            logger.info(f'''
#========================================================================
# CV: {cv_score} | Base CV: {base_cv_score}
# Update: {cv_score<base_cv_score}
# No Update Count: {no_update_cnt}
#========================================================================
''')

            # 半数以上のCVが更新されなかったら、途中でやめる
            if no_update_cnt >= int(len(seed_list)/2):
                logger.info(f'''
#========================================================================
# Not Update Over Half Seed: {no_update_cnt}>{int(len(seed_list)/2)}
#========================================================================
''')
                continue


        if path != 'base_path':
            train.drop(valid_feat, axis=1, inplace=True)
            #  test.drop(valid_feat, axis=1, inplace=True)

        df_pred['prediction'] /= len(seed_list)

        cv_score_mean = np.mean(cv_score_list)
        score_update = cv_score_mean < base_cv_score
        logger.info(f'''
#========================================================================
# CV Score Avg : {cv_score_mean} | Base Score: {base_cv_score}
# Score Update : {score_update}
# Valid Feature: {valid_feat}
#========================================================================
''')

        # UpdateされなかったFeatureは計算しない
        if not(score_update):
            continue

        #========================================================================
        # Result Summarize

        #========================================================================
        # outlierに対するスコアを出す
        out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
        out_score = np.sqrt(mean_squared_error(out_val, out_pred))
        LGBM.val_score_list.append(out_score)
        #========================================================================

        # 結果ファイルの作成
        LGBM.val_score_list.append(cv_score_mean)
        if i:
            feat_name = f"{i}_{valid_feat}"
        else:
            feat_name = f"{valid_feat}"
        tmp = pd.Series(LGBM.val_score_list, name=feat_name)
        valid_list.append(tmp.copy())
        tmp_valid_list.append(tmp.copy())
        if i==0:
            base_valid = tmp.copy()
            base_cv_score = cv_score_mean

        train_used_paths.append(path)
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
