import sys
is_linear = [True, False][int(sys.argv[1])]
out_part = ['', 'no_out'][1]
import gc
import re
import pandas as pd
import numpy as np
import os
import time
import datetime
import glob
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, reduce_mem_usage, elo_save_feature, impute_feature
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

#========================================================================
# Keras 
# Corporación Favorita Grocery Sales Forecasting
sys.path.append(f'{HOME}/kaggle/data_analysis/model')
from nn_keras import elo_build_NN, RMSE, elo_build_linear_NN
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#========================================================================


#========================================================================
# Args
out_part = ['', 'part', 'all'][0]
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term', 'no_out_flg']
stack_name='keras'
model_type='keras'
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
seed = 328
#========================================================================


#========================================================================
# Data Load 
print("Preparing dataset...")
# win_path = f'../features/4_winner/*.gz'
# Ensemble 1
set1 = f'../model/E1_set/*.gz'
# Ensemble 2
set2 = f'../model/E2_set/*.gz'
# Ensemble 3
set3 = f'../model/E3_set/*.gz'
# Ensemble 4
set4 = f'../model/E4_set/*.gz'

set_list = [set1, set2, set3, set4]
win_path = set_list[int(sys.argv[2])]

win_path_list = glob.glob(win_path)

base = utils.read_df_pkl('../input/base_term*0*')[[key, target, 'first_active_month']]
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
feature_list = utils.parallel_load_data(path_list=win_path_list)
df = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)

train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True , drop=True)

if out_part=='no_out':
    train = train[train[target]>-30]
#========================================================================

#========================================================================
# 正規化の前処理(Null埋め, inf, -infの処理) 
for col in train.columns:
    if col in ignore_list: continue
        
    train[col] = impute_feature(train, col)
    test[col] = impute_feature(test, col)
#========================================================================

# #========================================================================
# # inf check
# length = len(train)
# for col in train.columns:
#     tmp = train[col].dropna().shape[0]
#     if length - tmp>0:
#         print(col)
        
#     inf_max = train[col].max()
#     inf_min = train[col].min()
#     if inf_max==np.inf or inf_min==-np.inf:
#         print(col, inf_max, inf_min)
# #========================================================================

#========================================================================
# CVの準備
fold = 6
kfold = utils.read_pkl_gzip('../input/ods_kfold.gz')
use_cols = [col for col in train.columns if col not in ignore_list]
scaler = StandardScaler()

# なぜか一回目で終わらないことがあるので。。
try:
    scaler.fit(pd.concat([train[use_cols], test[use_cols]]))
except ValueError:
    inf_col_list = []
    for col in use_cols:

        inf_max = train[col].max()
        inf_min = train[col].min()
        if inf_max==np.inf or inf_min==-np.inf:
            inf_col_list.append(col)
    
    for col in inf_col_list:
        train[col] = impute_feature(train, col)
        test[col] = impute_feature(test, col)
    scaler.fit(pd.concat([train[use_cols], test[use_cols]]))
    
x_test = scaler.transform(test[use_cols])
Y = train[target]
y_min = Y.min()
if not(is_linear):
    Y = Y - y_min+1

# x_test = x_test.as_matrix()
# For LSTM
if not(is_linear):
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    
print(f"Train: {train.shape} | Test: {test.shape}") 
#========================================================================

#========================================================================
# NN Model Setting 
fold = 6
N_EPOCHS = 15
N_EPOCHS = 30
# batch_size = 65536
batch_size = 1024
# batch_size = 512
batch_size = 256
batch_size = 128
learning_rate = 1e-3
# learning_rate = 1e-2

if is_linear:
    model = elo_build_linear_NN(input_cols=len(use_cols))
else:    
#     model = elo_build_NN(input_rows=1, input_cols=len(use_cols))
    model = corp_1st_LSTM(input_rows=1, input_cols=len(use_cols))


opt = optimizers.Adam(lr=learning_rate)
model.compile(loss=RMSE, optimizer=opt, metrics=[RMSE])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
]
#========================================================================

#========================================================================
# Result Box
model_list = []
result_list = []
score_list = []
val_pred_list = []
test_pred = np.zeros(len(test))
#========================================================================

#========================================================================
# Train & Prediction Start

for fold_no, (trn_idx, val_idx) in enumerate(kfold):

    #========================================================================
    # Make Dataset
#     X_train, X_val = train_test_split(train, test_size=0.2)
    X_train, y_train = train.iloc[trn_idx, :][use_cols], Y.iloc[trn_idx]
    X_val, y_val = train.iloc[val_idx, :][use_cols], Y.iloc[val_idx]
    
     
    X_train[:] = scaler.transform(X_train)
    X_val[:] = scaler.transform(X_val)
    X_train = X_train.as_matrix()
    X_val = X_val.as_matrix()
    if not(is_linear):
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    #========================================================================
    
    # Fitting
    # なぜか平均を引いてる？そのほうがfitするの？
    # model.fit(X_train, y- y_mean, batch_size = batch_size, epochs = N_EPOCHS, verbose=2,
    #            validation_data=(X_val, y_val - y_mean), callbacks=callbacks )
    model.fit(X_train, y_train, batch_size = batch_size, epochs = N_EPOCHS, verbose=2,
               validation_data=(X_val, y_val), callbacks=callbacks )
    
    # Prediction
    y_pred = model.predict(X_val)
    y_pred = y_pred.reshape(y_pred.shape[0], )
    tmp_pred = model.predict(x_test)
    test_pred += tmp_pred.reshape(tmp_pred.shape[0], )
    
    if not(is_linear):
        y_val += (y_min-1)
        y_pred += (y_min-1)
        test_pred += (y_min-1)
#     model_list.append(model)
    
    # Stack Prediction
    df_pred = train.iloc[val_idx, :][[key, target]].copy()
    df_pred['prediction'] = y_pred
    result_list.append(df_pred)
    
    # Scoring
    err = (y_val - y_pred)
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'RMSE: {score} | SUM ERROR: {err.sum()}')
    score_list.append(score)
    #========================================================================

cv_score = np.mean(score_list)
logger.info(f'''
#========================================================================
# CV SCORE AVG: {cv_score}
#========================================================================''')

#========================================================================
# Stacking
test_pred /= fold
test['prediction'] = test_pred
stack_test = test[[key, 'prediction']]
result_list.append(stack_test)
df_pred = pd.concat(result_list, axis=0, ignore_index=True).drop(target, axis=1)
df_pred = base.merge(df_pred, how='inner', on=key)
print(f"Stacking Shape: {df_pred.shape}")

utils.to_pkl_gzip(obj=df_pred, path=f'../stack/{start_time[4:12]}_elo_NN_stack_linear{is_linear*1}_{len(use_cols)}feat_lr{learning_rate}_batch{batch_size}_epoch{N_EPOCHS}_CV{cv_score}')
#========================================================================