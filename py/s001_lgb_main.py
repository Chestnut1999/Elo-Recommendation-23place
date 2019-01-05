import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'weight']

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
    early_stopping_rounds = 100
num_boost_round = 30000

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
#  params['subsample'] = 0.7
#  params['lambda_l1'] = 0.0
#  params['colsample_bytree'] = 0.4
#  params['num_threads'] = 18

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path)
# tmp_path_listには検証中のfeatureを入れてある
tmp_path_list = glob.glob('../features/5_tmp/*.gz')
win_path_list += tmp_path_list

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

train_id = train[key].values
test_id = test[key].values

#========================================================================

#========================================================================
# LGBM Setting
try:
    seed_list = np.arange(int(sys.argv[4]))
    seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005]
except IndexError:
    seed_list = [1208]
metric = 'rmse'
#  metric = 'mse'
params['metric'] = metric
fold=5
fold_type='self'
group_col_name=''
dummie=1
oof_flg=True
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)

from sklearn.model_selection import StratifiedKFold

# seed_avg
seed_pred = np.zeros(len(test))
cv_list = []
for i, seed in enumerate(seed_list):

    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    LGBM.seed = seed

    train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    kfold = folds.split(train,train['outliers'].values)

    #  train['weight'] = train['outliers']*-0.2 + 1.0
    #  params['weight'] = 'weight'
    train.drop('outliers', axis=1, inplace=True)
#========================================================================

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
        ,self_kfold=kfold
        #  ,comp_name='elo'
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

cv_feim.to_csv(f'../valid/{start_time[4:12]}_{model_type}_{fname}_feat{feature_num}_CV{cv_score}_lr{learning_rate}.csv', index=False)

#========================================================================
# STACKING
if len(stack_name)>0:
    logger.info(f'result_stack shape: {df_pred.shape}')
    if len(seed_list)==1:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)
    else:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{len(seed_list)}seed_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)
        pred_cols = [col for col in df_pred.columns if col.count('predict')]
        df_pred['pred_mean'] = df_pred[pred_cols].mean(axis=1)
        df_pred['pred_std'] = df_pred[pred_cols].std(axis=1)
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{len(seed_list)}seed_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)


# outlierに対するスコアを出す
from sklearn.metrics import mean_squared_error
train.reset_index(inplace=True)
out_ids = train.loc[train.target<-30, key].values
out_val = train.loc[train.target<-30, target].values
if len(seed_list)==1:
    out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
else:
    out_pred = df_pred[df_pred[key].isin(out_ids)]['pred_mean'].values
out_score = np.sqrt(mean_squared_error(out_val, out_pred))
logger.info(f'''
#========================================================================
# OUTLIER FIT SCORE: {out_score}
#========================================================================''')


logger.info(f'FEATURE IMPORTANCE PATH: {HOME}/kaggle/home-credit-default-risk/output/cv_feature{feature_num}_importances_{metric}_{cv_score}.csv')

#========================================================================

#========================================================================
# Submission
if len(submit)>0:
    submit[target] = test_pred
    submit.to_csv(f'../submit/{start_time[4:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_{len(seed_list)}seed_CV{cv_score}_LB.csv', index=False)
#========================================================================


#========================================================================
# X-RAYの計算と出力
# Args:
#     model    : 学習済のモデル
#     train    : モデルの学習に使用したデータセット
#     col_list : X-RAYの計算を行うカラムリスト。指定なしの場合、
#                データセットの全カラムについて計算を行うが、
#                計算時間を考えると最大30カラム程度を推奨。
#========================================================================
if xray:
    # Squeeze card_id
    id_list = pd.read_csv('../output/1229_elo_id_list_std001_max-5.csv')[key].values
    train.reset_index(inplace=True)
    train = train.loc[train[key].isin(id_list), :]
    # ===

    train.reset_index(inplace=True)
    train = train[LGBM.use_cols]
    result_xray = pd.DataFrame()
    N_sample = 500000
    max_point = 30
    for fold_num in range(fold):
        if fold_num==0:
            xray_obj = Xray_Cal(logger=logger, ignore_list=ignore_list)
        xray_obj.model = LGBM.fold_model_list[fold_num]
        xray_obj, tmp_xray = xray_obj.get_xray(base_xray=train, col_list=train.columns, fold_num=fold_num, N_sample=N_sample, max_point=max_point, parallel=False)
        tmp_xray.rename(columns={'xray':f'xray_{fold_num}'}, inplace=True)

        if len(result_xray):
            result_xray = result_xray.merge(tmp_xray.drop('N', axis=1), on=['feature', 'value'], how='inner')
        else:
            result_xray = tmp_xray.copy()
        del tmp_xray
        gc.collect()

    xray_col = [col for col in result_xray.columns if col.count('xray')]
    result_xray['xray_avg'] = result_xray[xray_col].mean(axis=1)
    result_xray.to_csv(f'../output/{start_time[4:10]}_xray_{model_type}_CV{LGBM.cv_score}.csv', index=False)
