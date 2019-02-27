import gc
import re
import pandas as pd
import numpy as np
import os
import sys
import time
import datetime
import glob
sys.path.append('../py/')
from s027_kfold_ods import ods_kfold
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, reduce_mem_usage, elo_save_feature, impute_feature
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


try:
    debug = int(sys.argv[1])
except IndexError:
    debug = 0

#========================================================================
# Args
out_part = ['', 'part', 'all'][0]
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term', 'no_out_flg', 'clf_pred']
stack_name='ridge'
submit = pd.read_csv('../input/sample_submission.csv')
model_type='KNN'
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
seed = 328
#========================================================================


#========================================================================
# Path List 
def get_dataset(base, model_no):
    win_path = f'../features/4_winner/*.gz'
    #  win_path = f'../features/1_first_valid/*.gz'
    model_path_list = [f'../model/LB3670_70leaves_colsam0322/*.gz', '../model/E2_lift_set/*.gz', '../model/E3_PCA_set/*.gz', '../model/E4_mix_set/*.gz', '../model/LB3669LB_70leaves/*.gz'][model_no]
    model_path = model_path_list[model_no]
    tmp_path_list = glob.glob(f'../features/5_tmp/*.gz') + glob.glob(f'../features/0_exp/*.gz')
    #  tmp_path_list = glob.glob(f'../features/5_tmp/*.gz')
    win_path_list = glob.glob(model_path) + glob.glob(win_path) + tmp_path_list
    #  win_path_list = glob.glob(model_path) + tmp_path_list
    #  win_path_list = glob.glob(model_path) + glob.glob(win_path)
    win_path_list = glob.glob(win_path) + tmp_path_list
    #  win_path_list = glob.glob(model_path) + glob.glob(win_path) + tmp_path_list
    #========================================================================
    
    feature_list = utils.parallel_load_data(path_list=win_path_list)
    df_feat = pd.concat(feature_list, axis=1)
    base = pd.concat([base, df_feat], axis=1)
    
    train = base[~base[target].isnull()]
    test = base[base[target].isnull()]
    
    if debug:
        train = train.head(10000)
        test = test.head(1000)
    
    for col in train.columns:
        if col in ignore_list:
            continue
        train[col] = utils.impute_feature(df=train, col=col)
        test[col] = utils.impute_feature(df=test, col=col)
    
    return train, test



model_no = 0
base = utils.read_pkl_gzip('../input/base_type_group.gz')[[key, target]]
base_train, base_test = get_dataset(base, model_no)


#========================================================================
# Make Dataset 
pred_col = 'prediction'
valid_type = 'ods'
set_type = 'all'
#========================================================================
    
#========================================================================
# CVの準備
fold_seed = 328
fold = 6

#========================================================================
# Dataset
submit = pd.read_csv('../input/sample_submission.csv').set_index(key)
result_list = []
score_list = []
feat_list = [col for col in base_train.columns if col not in ignore_list]
#  use_cols = []
#  feim = pd.read_csv('../valid/0224_215_valid_lgb_lr0.01_272feats_10seed_70leaves_iter1161_OUT0_CV3.6176129805843_LB.csv')
#  top100 = feim['feature'].values[:100]
#  for col in top100:
#      for feat in feat_list:
#          if feat.count(col):
#              use_cols.append(col)
np.random.seed(1208)
np.random.shuffle(feat_list)
#========================================================================

if debug:
    use_cols = use_cols[:10]
train = base_train.copy()
test = base_test.copy()
Y = train[target]

#========================================================================
# NN Model Setting 
params = {}
params['n_jobs']=-1
params['n_neighbors']=350
# params['metric']='rmse'
model = KNeighborsRegressor(**params)

kfold = utils.read_pkl_gzip(f'../input/kfold_ods_equal_seed328.gz')

#========================================================================
# Preset
test_pred = np.zeros(len(test))
result_list = []
score_list = []
#========================================================================


for num in np.arange(10, 211, 10):
    #  use_cols = feat_list[30-num:num]
    use_cols = feat_list[num-10:num]

    #========================================================================
    # Train & Prediction Start
    for fold_no, (trn_idx, val_idx) in enumerate(zip(*kfold)):

        if key not in train.columns:
            train = train.reset_index()
            test = test.reset_index()

        #========================================================================
        # Make Dataset
        scaler = StandardScaler()
        scaler.fit(pd.concat([train[use_cols], test[use_cols]]))
        x_test = scaler.transform(test[use_cols])

        X_train, y_train = train.loc[train[key].isin(trn_idx), :][use_cols], Y.loc[train[key].isin(trn_idx)]
        X_val, y_val = train.loc[train[key].isin(val_idx), :][use_cols], Y.loc[train[key].isin(val_idx)]

        X_train[:] = scaler.transform(X_train)
        X_val[:] = scaler.transform(X_val)
        X_train = X_train.as_matrix()
        X_val = X_val.as_matrix()

        print(f"Train: {X_train.shape} | Valid: {X_val.shape} | Test: {x_test.shape}")
        #========================================================================

        # Fitting
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_val)
        test_pred += model.predict(x_test)

        df_pred = train.loc[train[key].isin(val_idx), :][[key, target]].copy()
        df_pred['prediction'] = y_pred
        result_list.append(df_pred)

        # Scoring
        score = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f'RMSE: {score}')
        score_list.append(score)
        #========================================================================

    cv_score = np.mean(score_list)

    #========================================================================
    # Stacking
    test_pred /= fold_no+1
    test['prediction'] = test_pred
    stack_test = test[[key, 'prediction']]

    result_list.append(stack_test)
    df_pred = pd.concat(result_list, axis=0, ignore_index=True).drop(target, axis=1)
    if key not in base:
        base.reset_index(inplace=True)
    df_pred = base[[key, target]].merge(df_pred, how='inner', on=key)

    print(f'''
    # =====================================================================
    #  SCORE AVG: {cv_score}
    # =====================================================================''')

    #========================================================================
    # Save Stack
    feature = df_pred['prediction'].values
    utils.to_pkl_gzip(path=f"../features/1_first_valid/{start_time[4:12]}_stack_{model_type}_set-{set_type}_valid-{valid_type}_seed{fold_seed}_feat{len(use_cols)}_CV{cv_score}_LB" , obj=feature)
    #========================================================================
