import sys
is_save = 0
out_part = ['all', 'no_out'][0]
model_type = ['ridge', 'rmf', 'ext']
try:
    model_no = int(sys.argv[6])
except IndexError:
    model_no = 0
    
import gc
import re
import pandas as pd
import numpy as np
import os
import time
import datetime
import glob
sys.path.append('../py/')
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
from utils import logger_func, get_categorical_features, get_numeric_features, reduce_mem_usage, elo_save_feature, impute_feature
import utils
try:
    if not logger:
        logger = logger_func()
except NameError:
    logger = logger_func()

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

#========================================================================
# Model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#========================================================================


# ========================================================================
# Args
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month',
               'index', 'personal_term', 'no_out_flg']
stack_name = model_type
submit = pd.read_csv('../input/sample_submission.csv')
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
# ========================================================================


# ========================================================================
# Data Load
print("Preparing dataset...")

model_path_list = [f'../model/LB3670_70leaves_colsam0322/*.gz', '../model/E2_lift_set/*.gz', '../model/E3_PCA_set/*.gz', '../model/E4_mix_set/*.gz']
model_path = model_path_list[model_no]
win_path_list = glob.glob(model_path)

base = utils.read_df_pkl(
    '../input/base_term*0*')[[key, target, 'first_active_month']]
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
feature_list = utils.parallel_load_data(path_list=win_path_list)
df = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df.iloc[:len(base_train), :]], axis=1)
test = pd.concat(
    [base_test, df.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)

train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

# ========================================================================

# ========================================================================
# 正規化の前処理(Null埋め, inf, -infの処理)
if model_type=='ridge':
    for col in train.columns:
        if col in ignore_list:
            continue

        train[col] = impute_feature(train, col)
        test[col] = impute_feature(test, col)
# ========================================================================

# ========================================================================
# CVの準備
seed = 328
num_threads = -1
fold = 6
fold_seed = 328
submit = pd.read_csv('../input/sample_submission.csv').set_index(key)
model_list = []
result_list = []
score_list = []
val_pred_list = []
test_pred = np.zeros(len(test))

ignore_list = [key, target, 'merchant_id', 'first_active_month',
               'index', 'personal_term', 'no_out_flg' 'clf_pred']
# ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term']
use_cols = [col for col in train.columns if col not in ignore_list]
scaler = StandardScaler()
scaler.fit(pd.concat([train[use_cols], test[use_cols]]))
x_test = scaler.transform(test[use_cols])

if out_part=='no_out':
    train = train[train[target]>-30]
    kfold = utils.read_pkl_gzip('../input/kfold_ods_no_out_fold6_seed328.gz')
else:
    kfold = utils.read_pkl_gzip('../input/kfold_ods_all_fold6_seed328.gz')
Y = train[target]
# ========================================================================

print(f"Train: {train.shape} | Test: {test.shape}")

# ========================================================================
# NN Model Setting

def select_model(model_type, seed=1208):
    if model_type=='ridge':
        params['solver'] = 'auto'
        params['fit_intercept'] = True
        params['alpha'] = 0.4
        params['max_iter'] = 1000
        params['normalize'] = False
        params['tol'] = 0.01
        model = Ridge(**params)
    elif model_type=='rmf':
        params['maxdepth'] = 10
        params['n_estimators'] = 3000
        params['criterion'] = 'mse'
        params['max_features'] = 0.3
        params['min_samples_leaf'] = 30
        params['min_samples_split'] = 30
        params['n_jobs'] = num_threads
        params['random_state'] = seed
        model = RandomForestRegressor(**params)
    elif model_type=='ext':
        params['maxdepth'] = 10
        params['n_estimators'] = 3000
        params['max_features'] = 'auto'
        params['min_samples_leaf'] = 30
        params['min_samples_split'] = 30
        params['n_jobs'] = num_threads
        params['random_state'] = seed
        model = ExtraTreesRegressor(**params)

    return model

result_list = []

# ========================================================================
# Train & Prediction Start
for fold_no, (trn_idx, val_idx) in enumerate(zip(*kfold)):
    if key not in train.columns:
        train = train[~train[target].isnull()].reset_index()
        test = test[test[target].isnull()].reset_index()

    # ========================================================================
    # Make Dataset
    X_train, y_train = train.loc[train[key].isin( trn_idx), :][use_cols], Y.loc[train[key].isin(trn_idx)]
    X_val, y_val = train.loc[train[key].isin( val_idx), :][use_cols], Y.loc[train[key].isin(val_idx)]

    X_train[:] = scaler.transform(X_train)
    X_val[:] = scaler.transform(X_val)
    X_train = X_train.as_matrix()
    X_val = X_val.as_matrix()
    # ========================================================================

    # Fitting
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_val)
    test_pred += model.predict(x_test)

    # Stack Prediction
    df_pred = train.loc[train[key].isin(val_idx), :][[key, target]].copy()
    df_pred['prediction'] = y_pred
    result_list.append(df_pred)

    # Scoring
    err = (y_val - y_pred)
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'RMSE: {score} | SUM ERROR: {err.sum()}')
    score_list.append(score)
    # ========================================================================

cv_score = np.mean(score_list)

# ========================================================================
# Stacking
test_pred /= fold_no+1
test['prediction'] = test_pred
stack_test = test[[key, 'prediction']]
result_list.append(stack_test)
df_pred = pd.concat(result_list, axis=0, ignore_index=True).drop(target, axis=1)

if key not in base:
    base.reset_index(inplace=True)
df_pred = base[[key, target]].merge(df_pred, how='inner', on=key)
print(f"Stacking Shape: {df_pred.shape}")
# ========================================================================

# ========================================================================
# outlierに対するスコアを出す
if key not in train.columns:
    train.reset_index(inplace=True)
out_ids = train.loc[train.target < -30, key].values
out_val = train.loc[train.target < -30, target].values
out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
out_score = np.sqrt(mean_squared_error(out_val, out_pred))
# ========================================================================

print(f'''
#========================================================================
# CV SCORE AVG: {cv_score}
# OUT SCORE: {out_score}
#========================================================================''')

# ========================================================================
# Submission
df_pred.set_index(key, inplace=True)
submit[target] = df_pred['prediction']
submit_path = f'../submit/{start_time[4:12]}_submit_RIDGE_STACKING_{model_type}_{len(use_cols)}models_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv'
display(submit.head())
# ========================================================================

# ========================================================================
# Save Result
if is_save:
    submit.to_csv(submit_path, index=True)
    utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_stack_{model_type}_alpha{alpha}_{len(use_cols)}feats_tol{tol}_iter{max_iter}_OUT{str(out_score)[:7]}_CV{str(cv_score).replace('.', '-')}_LB" , obj=df_pred)
#========================================================================
