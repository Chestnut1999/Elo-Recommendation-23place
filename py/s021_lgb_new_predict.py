num_threads = 36
import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term']

fname=''
xray=False
#  xray=True
submit = pd.read_csv('../input/sample_submission.csv')
#  submit = []


model_type='lgb'
try:
    learning_rate = float(sys.argv[1])
except ValueError:
    learning_rate = 0.01
early_stopping_rounds = 200
#  early_stopping_rounds = 150
num_boost_round = 5000

import numpy as np
import datetime
import glob
import gc
import os
from sklearn.metrics import mean_squared_error
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

# Best outlier fit LB3.690
#  num_leaves = 4
#  num_leaves = 16
num_leaves = 31
num_leaves = 48
num_leaves = 59
num_leaves = 61
num_leaves = 68
params['num_leaves'] = num_leaves
params['num_threads'] = num_threads
if num_leaves>65:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2755158
    params['min_child_samples'] = 37
    params['lambda_l2'] = 7

elif num_leaves>60:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2792
    params['min_child_samples'] = 59
    params['lambda_l2'] = 2

elif num_leaves>50:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.256142
    params['min_child_samples'] = 55
    params['lambda_l2'] = 3

elif num_leaves>40:
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

win_path = f'../features/4_winner/*.gz'
tmp_path_list = glob.glob(f'../features/5_tmp/*.gz') + glob.glob(f'../features/0_exp/*.gz')

base = utils.read_df_pkl('../input/base_first*')
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

win_path_list = glob.glob(win_path) + tmp_path_list
feature_list = utils.parallel_load_data(path_list=win_path_list)

df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)

target = '556_lif_hist_regist_term_lift_203_pst_ratio_new_auth1_purchase_amount_sum@'
stack_name = target
train_test = pd.concat([train, test], axis=0)
train = train_test[train_test[target]!=-1]
test = train_test[train_test[target]==-1]
test[target] = np.nan
train['target'] = train[target]
test['target'] = test[target]
target = 'target'

#========================================================================

#========================================================================
# LGBM Setting
try:
    argv3 = int(sys.argv[3])
    seed_list = np.arange(argv3)
    if argv3<=10:
        #  seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005, 2019][:argv3]
        seed_list = [328, 605, 1212, 1222, 405, 1128, 1012, 1208, 2005, 2019][:argv3]
except IndexError:
    seed_list = [1208]
    seed_list = [328]
metric = 'rmse'
#  metric = 'mse'
params['metric'] = metric
fold=5
fold_type='self'
#  fold_type='stratified'
group_col_name=''
dummie=1
oof_flg=True
LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)

train, test, drop_list = LGBM.data_check(train=train, test=test, target=target, encode='label')
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)

from sklearn.model_selection import StratifiedKFold, KFold


# seed_avg
seed_pred = np.zeros(len(test))
cv_list = []
iter_list = []
model_list = []
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True , drop=True)
for i, seed in enumerate(seed_list):

    if key not in train.columns:
        train.reset_index(inplace=True)
    if key not in test.columns:
        test.reset_index(inplace=True)

    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    LGBM.seed = seed


    #========================================================================
    # Validation Setting vvv
    if sys.argv[4]=='ods':

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
        kfold = folds.split(train, train[target].values)


    # 3. Default KFold
    else:
        kfold = False
        fold_type = 'kfold'
    #========================================================================

    train.sort_index(axis=1, inplace=True)
    test.sort_index(axis=1, inplace=True)

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
        iter_list.append(LGBM.iter_avg)
        cv_feim = LGBM.cv_feim
        feature_num = len(LGBM.use_cols)
        df_pred = LGBM.result_stack.copy()
    else:
        cv_score = LGBM.cv_score
        cv_list.append(cv_score)
        iter_list.append(LGBM.iter_avg)
        LGBM.cv_feim.columns = [col if col.count('feature') else f"{col}_{seed}" for col in LGBM.cv_feim.columns]
        cv_feim = cv_feim.merge(LGBM.cv_feim, how='inner', on='feature')
        df_pred = df_pred.merge(LGBM.result_stack[[key, 'prediction']].rename(columns={'prediction':f'prediction_{i}'}), how='inner', on=key)


#========================================================================
# STACKING
if len(stack_name)>0:
    logger.info(f'result_stack shape: {df_pred.shape}')
    if len(seed_list)>1:
        pred_cols = [col for col in df_pred.columns if col.count('predict')]
        df_pred['pred_mean'] = df_pred[pred_cols].mean(axis=1)
        df_pred['pred_std'] = df_pred[pred_cols].std(axis=1)
#========================================================================


#========================================================================
# Result
cv_score = np.mean(cv_list)
iter_avg = np.int(np.mean(iter_list))
#========================================================================

logger.info(f'''
#========================================================================
# {len(seed_list)}SEED CV SCORE AVG: {cv_score}
#========================================================================''')

# Save
try:
    if int(sys.argv[2])==0:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_stack_pred_{stack_name}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_CV{str(cv_score).replace('.', '-')}", obj=df_pred)
except ValueError:
    pass
except TypeError:
    pass
