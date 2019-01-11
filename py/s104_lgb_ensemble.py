import sys
import pandas as pd
#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id']

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
num_boost_round = 10000

import numpy as np
import datetime
import glob
import os
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from params_lgbm import params_elo
from xray_wrapper import Xray_Cal
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
    params = params_elo()
    params['learning_rate'] = learning_rate

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load
base = utils.read_df_pkl('../input/base*')
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

train_id = base_train[key].values
test_id = base_test[key].values

win_path10 = f'../features/4_winner/10*train*.gz'
win_path90 = f'../features/4_winner/90*train*.gz'
win_path10_list = glob.glob(win_path10)
win_path90_list = glob.glob(win_path90)

i = int(sys.argv[3])
np.random.seed(i)
#  np.random.seed(i+1208+605+2018)

use_feature10_path = list(np.random.choice(win_path10_list, 100, replace=False))
use_feature90_path = list(np.random.choice(win_path90_list, 700, replace=False))
train_feature_path = use_feature10_path + use_feature90_path

test_feature_path = []
for path in train_feature_path:
    test_feature_path.append(path.replace('train', 'test'))

train_feature_list = utils.pararell_load_data(path_list=train_feature_path)
test_feature_list = utils.pararell_load_data(path_list=test_feature_path)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)
test = pd.concat(test_feature_list, axis=1)
test = pd.concat([base_test, test], axis=1)

if i%10==0:
    outlier_pred = utils.read_pkl_gzip('../stack/1204_211_outlier_classify_lgb_auc0-8952469653357074_227features.gz').set_index(key)
    train['outlier_pred@'] = outlier_pred.loc[train_id, 'prediction'].values
    test['outlier_pred@'] = outlier_pred.loc[test_id, 'prediction'].values

# =====================================================================

# ========================================================================
# LGBM Setting
metric = 'rmse'
fold=5
fold_type='kfold'
group_col_name=''
dummie=1
oof_flg=True
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)

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
)

#========================================================================
# Result
#========================================================================
cv_score = LGBM.cv_score
test_pred = LGBM.prediction
cv_feim = LGBM.cv_feim
feature_num = len(LGBM.use_cols)

cv_feim.to_csv(f'../valid/{start_time[4:12]}_{model_type}_{fname}_feat{feature_num}_CV{cv_score}_lr{learning_rate}.csv', index=False)

#========================================================================
# STACKING
if len(stack_name)>0:
    logger.info(f'result_stack shape: {LGBM.result_stack.shape}')
    utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=LGBM.result_stack)
logger.info(f'FEATURE IMPORTANCE PATH: {HOME}/kaggle/home-credit-default-risk/output/cv_feature{feature_num}_importances_{metric}_{cv_score}.csv')
#========================================================================

#========================================================================
# Submission
if len(submit)>0:
    submit[target] = test_pred
    submit.to_csv(f'../submit/{start_time[4:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_CV{cv_score}_LB.csv', index=False)
#========================================================================
