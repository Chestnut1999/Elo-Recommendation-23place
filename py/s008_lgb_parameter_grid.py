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

try:
    params['num_threads'] = int(sys.argv[5])
except IndexError:
    params['num_threads'] = -1


# Best outlier fit LB3.690
params['subsample'] = 0.8757099996397999
params['colsample_bytree'] = 0.7401342964627846
params['num_leaves'] = 48
params['min_child_samples'] = 61

from sklearn.model_selection import ParameterGrid


# Ready Parmas

reg_sqrt = {'reg_sqrt':True}
objective_list = ['quantile', 'gamma', 'tweedie']
alpha = {"alpha":[0.2, 0.4, 0.6, 0.8]}
param_candidate = {'num_leaves': [4,6,8,10,12,14,16], 'max_depth': [3, 4]}
param_grid = list(ParameterGrid(param_candidate))

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


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#  scaler.fit_transform(train[target].values.reshape(-1,1))
train[target] = scaler.fit_transform(train[target].values.reshape(-1,1)).ravel()

# Exclude Difficult Outlier
#  clf_result = utils.read_pkl_gzip('../stack/0106_125_outlier_classify_9seed_lgb_binary_CV0-9045159588642034_179features.gz')[[key, 'prediction']]
#  train = train.merge(clf_result, how='inner', on=key)
#  tmp1 = train[train.prediction>0.05]
#  tmp2 = train[train.prediction<0.05][train.target>-30]
#  train = pd.concat([tmp1, tmp2], axis=0)
#  del tmp1, tmp2
#  gc.collect()
#  train.drop('prediction', axis=1, inplace=True)

#========================================================================

#========================================================================
# LGBM Setting
metric = 'rmse'
params['metric'] = metric
seed = 1208
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


# outlierに対するスコアを出す前準備
from sklearn.metrics import mean_squared_error
out_ids = train.loc[train.target<-30, key].values
out_val = train.loc[train.target<-30, target].values

cv_list = []
out_list = []

from sklearn.model_selection import StratifiedKFold
for i, obj in enumerate(objective_list):
#  for valid_param in param_grid:

    #  params.update(valid_param)
    params['objective'] = obj

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
        ,scaler=scaler
    )

    df_pred = LGBM.result_stack.copy()
    cv_score = LGBM.cv_score

    out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
    out_score = np.sqrt(mean_squared_error(out_val, out_pred))

    cv_list.append(cv_score)
    out_list.append(out_score)

    logger.info(f'''
    #========================================================================
    # OUTLIER FIT SCORE: {out_score}
    #========================================================================''')

#========================================================================
# Result
#========================================================================
feature_num = len(LGBM.use_cols)

logger.info(f'''
#========================================================================
# OUTLIER FIT SCORE:
{pd.DataFrame(np.asarray([cv_list, out_list]), columns=['cv_score', 'out_score'])}
#========================================================================''')

#========================================================================
# STACKING
#  if len(stack_name)>0:
#      logger.info(f'result_stack shape: {df_pred.shape}')
#      if len(seed_list)>1:
#          pred_cols = [col for col in df_pred.columns if col.count('predict')]
#          df_pred['pred_mean'] = df_pred[pred_cols].mean(axis=1)
#          df_pred['pred_std'] = df_pred[pred_cols].std(axis=1)


# Save
#  utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_stack_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_OUT{str(out_score)[:7]}_CV{str(cv_score).replace('.', '-')}_LB", obj=df_pred)
#  cv_feim.to_csv( f'../valid/{start_time[4:12]}_valid_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv' , index=False)

#========================================================================
# Submission
#  if len(submit)>0:
#      submit[target] = test_pred
#      submit_path = f'../submit/{start_time[4:12]}_submit_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv'
#      submit.to_csv(submit_path, index=False)
#========================================================================

