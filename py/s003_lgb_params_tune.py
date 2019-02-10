import optuna
import sys
import pandas as pd

# ========================================================================
# Args
# ========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index']

win_path = f'../features/4_winner/*.gz'

model_type='lgb'
learning_rate = 0.01
early_stopping_rounds = 200
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

params = params_elo()[1]
params['learning_rate'] = learning_rate

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load

win_path = f'../features/4_winner/*.gz'
model_path = f'../model/LB3670_70leaves_colsam0322/*.gz'
tmp_path_list = glob.glob(f'../features/5_tmp/*.gz') + glob.glob(f'../features/0_exp/*.gz')

base = utils.read_df_pkl('../input/base_first*')
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

win_path_list = glob.glob(win_path) + tmp_path_list
win_path_list = glob.glob(model_path) + glob.glob(win_path)
feature_list = utils.parallel_load_data(path_list=win_path_list)

df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)

train_id = train[key].values
#========================================================================


#========================================================================
# LGBM Setting
seed = 1208
seed = 328
metric = 'rmse'
fold=6
fold_type='self'
group_col_name=''
dummie=1
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, _, drop_list = LGBM.data_check(train=train, test=[], target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)

from sklearn.model_selection import StratifiedKFold, KFold

feat_list = glob.glob('../features/1_first_valid/*.gz')

valid_list = []

LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
LGBM.seed = seed

#========================================================================
# ods KFold 
train['rounded_target'] = train['target'].round(0)
train = train.sort_values('rounded_target').reset_index(drop=True)
vc = train['rounded_target'].value_counts()
vc = dict(sorted(vc.items()))
df = pd.DataFrame()
train['indexcol'],idx = 0,1
for k,v in vc.items():
    step = train.shape[0]/v
    indent = train.shape[0]/(v+1)
    df2 = train[train['rounded_target'] == k].sample(v, random_state=seed).reset_index(drop=True)
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

    #  subsample = trial.suggest_uniform('subsample', 0.9, 0.98)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.20, 0.33)
    num_leaves = trial.suggest_int('num_leaves', 54, 73)
    #  max_depth = trial.suggest_int('max_depth', 8, 12)
    min_child_samples = trial.suggest_int('min_child_samples', 30, 75)
    lambda_l2 = trial.suggest_int('lambda_l2', 3.0, 15.0)

    params = {
        'num_threads': -1,
        'num_leaves': num_leaves,
        'objective':'regression',
        "boosting": "gbdt",
        #  'max_depth': max_depth,
        'max_depth': -1,
        'learning_rate': learning_rate,
        "min_child_samples": min_child_samples,
        "bagging_freq": 1,
        #  "subsample": subsample ,
        "subsample": 0.9 ,
        "colsample_bytree": colsample_bytree,
        #  "colsample_bytree": 0.9,
        "metric": 'rmse',
        "lambda_l1": 0.1,
        "lambda_l2": lambda_l2,
        #  "lambda_l2": 0.1,
        "verbosity": -1,
        'random_seed': seed,
        'bagging_seed':seed,
        'feature_fraction_seed':seed,
        'data_random_seed':seed
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
    #  from sklearn.metrics import mean_squared_error
    #  train.reset_index(inplace=True)

    #  out_ids = train.loc[train.target<-30, key].values
    #  out_val = train.loc[train.target<-30, target].values
    #  out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
    #  out_score = np.sqrt(mean_squared_error(out_val, out_pred))

    #  out_list.append(out_score)
    #  if len(out_list)%10==0:
    #      if len(out_list)>=10:
    #          print(out_list[-10:])
    #      else:
    #          print(out_list)

    #  # outlier以外に対するスコアを出す
    #  com_ids = train.loc[train.target>-30, key].values
    #  com_val = train.loc[train.target>-30, target].values
    #  com_pred = df_pred[df_pred[key].isin(com_ids)]['prediction'].values
    #  com_score = np.sqrt(mean_squared_error(com_val, com_pred))
    #  com_list.append(com_score)
    #  com_score -= 1.8404775225287757

    logger.info(f'''
    #========================================================================
    # CV SCORE: {cv_score}
    #========================================================================''')
    #  if com_score<0:
    #      out_score += com_score*-2

    #  # スコア経過のログ
    #  LGBM.val_score_list.append(cv_score)
    #  LGBM.val_score_list.append(params)
    #  tmp = pd.Series(LGBM.val_score_list)
    #  valid_list.append(tmp.copy())

    return cv_score
    #  return out_score


study = optuna.create_study()
study.optimize(objective, n_trials=250)

df_valid = pd.concat(valid_list, axis=1)
df_valid.to_csv(f'../output/{start_time[4:11]}_elo_params_tune.csv')

out = pd.Series(out_list, name='OUT_SCORE').to_frame()
out['COM_SCORE'] = com_list
out.to_csv(f'../output/{start_time[4:11]}_elo_out_score_process.csv', index=True)
