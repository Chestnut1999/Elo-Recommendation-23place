import optuna
import sys
import pandas as pd

# ========================================================================
# Args
# ========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id']

win_path = f'../features/4_winner/*.gz'

# ========================================================================
# argv[1] : model_type 
# argv[2] : learning_rate
# argv[3] : early_stopping_rounds
# ========================================================================

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
for path in win_path_list:
    if path.count('train'):
        train_path_list.append(path)

base_train = base[~base[target].isnull()].reset_index(drop=True)
train_feature_list = utils.parallel_load_data(path_list=train_path_list)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)

train_id = train[key].values

#========================================================================

#========================================================================
# LGBM Setting
seed = 1208
metric = 'rmse'
fold=5
fold_type='self'
group_col_name=''
dummie=1
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, _, drop_list = LGBM.data_check(train=train, test=[], target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)

from sklearn.model_selection import StratifiedKFold

feat_list = glob.glob('../features/1_first_valid/*.gz')

valid_list = []

LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
LGBM.seed = seed

train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
kfold = folds.split(train,train['outliers'].values)
kfold = list(kfold)
train.drop('outliers', axis=1, inplace=True)


is_thres = False
if is_thres:
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
        ,params_tune=True
    )
    thres_score_list = LGBM.val_score_list
com_list = []
out_list = []

def objective(trial):

    subsample = trial.suggest_uniform('subsample', 0.5, 0.9)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.3, 0.85)
    num_leaves = trial.suggest_int('num_leaves', 18, 51)
    #  max_depth = trial.suggest_int('max_depth', 5, 10)
    min_child_samples = trial.suggest_int('min_child_samples', 20, 150)
    #  lambda_l2 = trial.suggest_int('lambda_l2', 0.0, 0.3)

    params = {
        'num_threads': -1,
        'num_leaves': num_leaves,
        'objective':'regression',
        "boosting": "gbdt",
        #  'max_depth': max_depth,
        'max_depth': -1,
        'learning_rate': 0.01,
        "min_child_samples": min_child_samples,
        "bagging_freq": 1,
        "subsample": subsample ,
        "colsample_bytree": colsample_bytree,
        #  "colsample_bytree": 0.9,
        "metric": 'rmse',
        "lambda_l1": 0.1,
        #  "lambda_l2": lambda_l2,
        "lambda_l2": 0.1,
        "verbosity": -1,
        'random_seed': 1208,
        'bagging_seed':1208,
        'feature_fraction_seed':1208,
        'data_random_seed':1208
    }

    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    LGBM.seed = seed

    #  train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
    #  folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    #  kfold = folds.split(train,train['outliers'].values)
    #  train.drop('outliers', axis=1, inplace=True)

    #========================================================================
    # Train & Prediction Start
    #========================================================================
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
        #  ,self_stop=thres_score_list
        ,params_tune=True
    )

    cv_score = LGBM.cv_score
    pred_val = LGBM.prediction
    df_pred = train.reset_index()[key].to_frame()
    df_pred['prediction'] = pred_val

    # outlierに対するスコアを出す
    from sklearn.metrics import mean_squared_error
    train.reset_index(inplace=True)

    out_ids = train.loc[train.target<-30, key].values
    out_val = train.loc[train.target<-30, target].values
    out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
    out_score = np.sqrt(mean_squared_error(out_val, out_pred))

    out_list.append(out_score)
    if len(out_list)%10==0:
        if len(out_list)>=10:
            print(out_list[-10:])
        else:
            print(out_list)

    # outlier以外に対するスコアを出す
    com_ids = train.loc[train.target>-30, key].values
    com_val = train.loc[train.target>-30, target].values
    com_pred = df_pred[df_pred[key].isin(com_ids)]['prediction'].values
    com_score = np.sqrt(mean_squared_error(com_val, com_pred))
    com_list.append(com_score)
    com_score -= 1.8404775225287757

    logger.info(f'''
    #========================================================================
    # COM SCORE: {com_score}
    # OUT SCORE: {out_score}
    #========================================================================''')
    if com_score<0:
        out_score += com_score*-2

    # スコア経過のログ
    LGBM.val_score_list.append(cv_score)
    LGBM.val_score_list.append(params)
    tmp = pd.Series(LGBM.val_score_list)
    valid_list.append(tmp.copy())

    #  return cv_score
    return out_score


study = optuna.create_study()
study.optimize(objective, n_trials=200)

df_valid = pd.concat(valid_list, axis=1)
df_valid.to_csv(f'../output/{start_time[4:11]}_elo_params_tune.csv')

out = pd.Series(out_list, name='OUT_SCORE').to_frame()
out['COM_SCORE'] = com_list
out.to_csv(f'../output/{start_time[4:11]}_elo_out_score_process.csv', index=True)
