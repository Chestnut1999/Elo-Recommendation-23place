import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month']

win_path = f'../features/4_winner/*.gz'
stack_name='en_route'
fname=''
xray=False
#  xray=True
submit = pd.read_csv('../input/sample_submission.csv')
#  submit = []

import numpy as np
import datetime
import glob
import gc
import os
HOME = os.path.expanduser('~')

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path)
train_path_list = []
test_path_list = []
for path in win_path_list:
    if path.count('train'):
        train_path_list.append(path)
    elif path.count('test'):
        test_path_list.append(path)


make_input = True
#  make_input = False
if make_input:

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

    y = train[[key, target]]
    y[target] = y[target].map(lambda x: 1 if x<-30 else 0)
    train.drop(target, axis=1)
    train = train.merge(y, how='inner', on=key).drop(key, axis=1)
    train.to_csv('../input/ffm_train.csv', index=False)
    sys.exit()


#========================================================================

import xlearn as xl
# Training task
ffm_model = xl.create_ffm()  # Use field-aware factorization machine
ffm_model.setTrain("../input/ffm_train.csv")   # Training data
#  ffm_model.setValidate("../input/titanic_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}

# Train model
ffm_model.fit(param, "../model.out")

sys.exit()

#========================================================================
# LGBM Setting
try:
    seed_list = np.arange(int(sys.argv[4]))
    seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005]
except IndexError:
    seed_list = [1208]
metric = 'rmse'
fold=5
fold_type='self'
group_col_name=''
dummie=1
oof_flg=True


#========================================================================
# Result
#========================================================================
test_pred = seed_pred / len(seed_list)

cv_feim.to_csv(f'../valid/{start_time[4:12]}_{model_type}_{fname}_feat{feature_num}_CV{cv_score}_lr{learning_rate}.csv', index=False)

#========================================================================
# STACKING
if len(stack_name)>0:
    logger.info(f'result_stack shape: {df_pred.shape}')
    if len(seed_list)==1:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)
    else:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{len(seed_list)}seed_{model_type}_CV{str(cv_score).replace('.', '-')}_{feature_num}features", obj=df_pred)
logger.info(f'FEATURE IMPORTANCE PATH: {HOME}/kaggle/home-credit-default-risk/output/cv_feature{feature_num}_importances_{metric}_{cv_score}.csv')
#========================================================================

#========================================================================
# Submission
if len(submit)>0:
    submit[target] = test_pred
    submit.to_csv(f'../submit/{start_time[4:12]}_submit_{model_type}_rate{learning_rate}_{feature_num}features_{len(seed_list)}seed_CV{cv_score}_LB.csv', index=False)
#========================================================================

