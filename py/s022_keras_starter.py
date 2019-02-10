debug = True
import gc
import re
import pandas as pd
import numpy as np
import os
import sys
import time
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
from nn_keras import elo_build_NN
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
submit = pd.read_csv('../input/sample_submission.csv')
model_type='keras'
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
#========================================================================


#========================================================================
# Data Load 
print("Preparing dataset...")
win_path = f'../features/4_winner/*.gz'
win_path = f'../model/LB3670_70leaves_colsam0322/*.gz'
# win_path_list = glob.glob(win_path) + glob.glob('../features/5_tmp/*.gz')
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

if debug:
    train = train.head(10000)
    test = test.head(2000)
#========================================================================

#========================================================================
# 正規化の前処理(Null埋め, inf, -infの処理) 
for col in train.columns:
    if col in ignore_list: continue
        
    train[col] = impute_feature(train, col)
    test[col] = impute_feature(test, col)
#========================================================================

#========================================================================
# ods.ai 3rd kernel
# https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78903
# KFold: n_splits=6(or 7)!, shuffle=False!
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
# =======================================================================

#========================================================================
# CVの準備
result_list = []
score_list = []
val_pred_list = []
test_pred = np.zeros(len(test))

N_EPOCHS = 30
# batch_size = 65536
batch_size = 128
learning_rate = 0.001

use_cols = [col for col in train.columns if col not in ignore_list]
scaler = StandardScaler()
scaler.fit(pd.concat([train[use_cols], test[use_cols]]))
test[:] = scaler.transform(test[use_cols])
test = test.as_matrix()
test = test.reshape((test.shape[0], 1, test.shape[1]))

Y = train[target]
y_mean = Y.mean()
#========================================================================
    
#========================================================================
# NN Model Setting 
model = build_model()
opt = optimizers.Adam(lr=learning_rate)
model.compile(loss=RMSE, optimizer=opt, metrics=[RMSE])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
]
#========================================================================


#========================================================================
# Train & Prediction Start

for fold_no, (trn_idx, val_idx) in kfold:

    #========================================================================
    # Make Dataset
#     X_train, X_val = train_test_split(train, test_size=0.2)
    X_train, y_train = train.iloc[trn_idx, :][use_cols], Y.iloc[trn_idx]
    X_val, y_val = train.iloc[val_idx, :][use_cols], Y.iloc[val_idx]
    
     
    X_train[:] = scaler.transform(X_train)
    X_val[:] = scaler.transform(X_val)
    X_train = X_train.as_matrix()
    X_val = X_val.as_matrix()
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
    y_pred = y_pred.reshape(y_pred.shape[1], )
    tmp_pred = model.predict(test)
    test_pred += test_pred.reshape(test_pred.shape[1], )
    
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

utils.to_pkl_gzip(obj=df_pred, path=f'../output/{start_time[4:11]}_elo_NN_stack_CV{score}')
#========================================================================


sys.exit()

#========================================================================
# Part of card_id Score
bench = pd.read_csv('../input/bench_LB3684_FAM_cv_score.csv')
part_score_list = []
part_N_list = []
fam_list = []
#  for i in range(201101, 201713, 1):
for i in range(201501, 201713, 1):
    fam = str(i)[:4] + '-' + str(i)[-2:]
    df_part = base_train[base_train['first_active_month']==fam]
    if len(df_part)<1:
        continue
    part_id_list = df_part[key].values

    part_train = df_pred.loc[df_pred[key].isin(part_id_list), :]
    y_train = part_train[target].values
    if 'pred_mean' in list(part_train.columns):
        y_pred = part_train['pred_mean'].values
    else:
        y_pred = part_train['prediction'].values

    y_pred = np.where(y_pred != y_pred, 0, y_pred)
    # RMSE
    part_score = np.sqrt(mean_squared_error(y_train, y_pred))
    bench_score = bench[bench['FAM']==fam]['CV'].values[0]
    part_score -= bench_score

    fam_list.append(fam)
    part_score_list.append(part_score)
    part_N_list.append(len(part_id_list))

#  for i, part_score, N in zip(fam_list, part_score_list, part_N_list):
df = pd.DataFrame(np.asarray([fam_list, part_score_list, part_N_list]).T)
df.columns = ['FAM', 'CV', 'N']

# FAM: {i} | CV: {part_score} | N: {len(part_id_list)}
pd.set_option('max_rows', 200)
logger.info(f'''
#========================================================================
# {df}
#========================================================================''')
#========================================================================


if len(train)>150000:
    if len(train[train[target]<-30])>0:
        # outlierに対するスコアを出す
        train.reset_index(inplace=True)
        out_ids = train.loc[train.target<-30, key].values
        out_val = train.loc[train.target<-30, target].values
        if len(seed_list)==1:
            out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
        else:
            out_pred = df_pred[df_pred[key].isin(out_ids)]['pred_mean'].values
        out_score = np.sqrt(mean_squared_error(out_val, out_pred))
    else:
        out_score = 0
else:
    out_score = 0

# Save
try:
    if int(sys.argv[2])==0:
        utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_stack_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_OUT{str(out_score)[:7]}_CV{str(cv_score).replace('.', '-')}_LB", obj=df_pred)
except ValueError:
    pass
except TypeError:
    pass

# 不要なカラムを削除
drop_feim_cols = [col for col in cv_feim.columns if col.count('importance_') or col.count('rank_')]
cv_feim.drop(drop_feim_cols, axis=1, inplace=True)
drop_feim_cols = [col for col in cv_feim.columns if col.count('importance') and not(col.count('avg'))]
cv_feim.drop(drop_feim_cols, axis=1, inplace=True)
cv_feim.to_csv( f'../valid/{start_time[4:12]}_valid_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv' , index=False)

#========================================================================
# Submission
try:
    if int(sys.argv[2])==0:
        test_pred = seed_pred / len(seed_list)
        submit[target] = test_pred
        submit_path = f'../submit/{start_time[4:12]}_submit_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv'
        submit.to_csv(submit_path, index=False)
except ValueError:
    pass
except TypeError:
    pass
#========================================================================

#========================================================================
# CV INFO

try:
    if int(sys.argv[2])==0 and len(train)>150000:

        import re
        path_list = glob.glob('../log_submit/0*CV*LB*.csv')
        path_list.append(submit_path)
        #  path_list_2 = glob.glob('../check_submit/*.csv')
        #  path_list += path_list_2

        tmp_list = []
        path_list = list(set(path_list))
        for path in path_list:
            tmp = pd.read_csv(path)
            tmp_path = path.replace(".", '-')
            cv = re.search(r'CV([^/.]*)_LB', tmp_path).group(1).replace('-', '.')
            lb = re.search(r'LB([^/.]*).csv', tmp_path).group(1).replace('-', '.')
            #  if lb<'3.690' and path!=submit_path:
            #      continue
            tmp.rename(columns={'target':f"CV{cv[:9]}_LB{lb}"}, inplace=True)
            tmp.set_index('card_id', inplace=True)
            tmp_list.append(tmp.copy())

        if len(tmp_list)>0:
            df = pd.concat(tmp_list, axis=1)
            df_corr = df.corr(method='pearson')

            logger.info(f'''
#========================================================================
# OUTLIER FIT SCORE: {out_score}
# SUBMIT CORRELATION:
{df_corr[f'CV{str(cv_score)[:9]}_LB'].sort_values()}
#========================================================================''')
except ValueError:
    pass
except TypeError:
    pass
