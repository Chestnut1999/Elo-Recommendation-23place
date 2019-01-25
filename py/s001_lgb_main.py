out_part = ['', 'part', 'all'][2]
#  num_threads = 32
num_leaves = 31
#  num_leaves = 16
#  num_leaves = 48
import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'column_0']

stack_name='en_route'
fname=''
xray=False
#  xray=True
submit = pd.read_csv('../input/sample_submission.csv')
#  submit = []


model_type='lgb'
try:
    learning_rate = float(sys.argv[1])
except IndexError:
    learning_rate = 0.1
early_stopping_rounds = 150
num_boost_round = 5000
#  try:
#      num_threads = int(sys.argv[6])
#  except IndexError:
#      num_threads = -1

import numpy as np
import datetime
import glob
import gc
import os
from sklearn.metrics import mean_squared_error
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

# Best outlier fit LB3.690
#  num_leaves = 4
params['num_leaves'] = num_leaves
if num_leaves>40:
    params['num_leaves'] = num_leaves
    params['subsample'] = 0.8757099996397999
    params['colsample_bytree'] = 0.7401342964627846
    params['min_child_samples'] = 61


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Data Load

try:
    # FMMのwinnerを使う時
    if int(sys.argv[2])>0:
        win_path = f'../model/2017{sys.argv[2]}/4_winner/*.gz'
        tmp_path_list = glob.glob('../model/2017{sys.argv[2]}/5_tmp/*.gz')
    # features配下のwinnerを使う時
    else:
        win_path = f'../features/4_winner/*.gz'
        tmp_path_list = glob.glob('../features/5_tmp/*.gz')
        # features配下のセットを使いつつ、first_monthを絞りたい場合
        if int(sys.argv[2])<0:
            if int(sys.argv[2])<10:
                sys.argv[2] = '0' + str(int(sys.argv[2])*-1)
            else:
                sys.argv[2] = str(int(sys.argv[2])*-1)
except ValueError:
    # ALLのwinnerを使う時
    if sys.argv[2]=='all':
        win_path = f'../model/all/4_winner/*.gz'
        tmp_path_list = glob.glob(f'../model/all/5_tmp/*.gz')
    else:
        # スードラベリングをするとき
        if sys.argv[2][-2:]=='pl':
            win_path = f'../model/2017{sys.argv[2][:2]}/{sys.argv[2][2:-2]}/*.gz'
            tmp_path_list = glob.glob(f'../model/2017{sys.argv[2][:2]}/5_tmp/*.gz')
        # スードラベリングをしないとき
        else:
            win_path = f'../model/2017{sys.argv[2][:2]}/{sys.argv[2][2:]}/*.gz'
            #  tmp_path_list = glob.glob(f'../model/2017{sys.argv[2][:2]}/5_tmp/*.gz')
            tmp_path_list = glob.glob(f'../features/exp/*.gz')


base = utils.read_df_pkl('../input/base_first*')
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

win_path_list = glob.glob(win_path) + tmp_path_list
feature_list = utils.parallel_load_data(path_list=win_path_list)
df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
# スードラベリングの時はtrainとtestのconcatした長さになる
pl_length = 0

#========================================================================
# card_id list by first active month
try:
    if int(sys.argv[2][:2])>0:
        train_latest_id_list = np.load(f'../input/card_id_train_first_active_2017{sys.argv[2][:2]}.npy')
        test_latest_id_list = np.load(f'../input/card_id_test_first_active_2017{sys.argv[2][:2]}.npy')

        if sys.argv[2][-2:]=='pl':
            pred_path = glob.glob(f'../model/2017{sys.argv[2][:2]}/stack/*org0_*')[0]
            pred_col = 'pred'
            pred_feat = utils.read_pkl_gzip(pred_path)
            train[pred_col] = pred_feat[:len(train)]
            train.loc[~train[key].isin(train_latest_id_list), target] = train.loc[~train[key].isin(train_latest_id_list), pred_col]

            tmp_test = test.copy()
            tmp_test[target] = pred_feat[len(train):]

            # first_active_monthが201712より前の場合、学習データセットから未来のfirst_active_monthを除外する
            if int(sys.argv[2][:2])<12:
                base = base[base['first_active_month'] <= f'2017-{sys.argv[2][:2]}']
                train = train.merge(base[key].to_frame(), how='inner', on=key)
                test = test.merge(base[key].to_frame(), how='inner', on=key)
                tmp_test = tmp_test.merge(base[key].to_frame(), how='inner', on=key)


            train = pd.concat([train, tmp_test], axis=0, ignore_index=True).drop(pred_col, axis=1)
            pl_length = len(train)
            del tmp_test
            gc.collect()
        else:
            #  train = train.loc[train[key].isin(train_latest_id_list), :].reset_index(drop=True)
            #  test = test.loc[test[key].isin(test_latest_id_list), :].reset_index(drop=True)
            #  base = base[base['first_active_month'] <= f'2017-{sys.argv[2][:2]}']
            #  train = train.merge(base[key].to_frame(), how='inner', on=key)
            #  test = test.merge(base[key].to_frame(), how='inner', on=key)
            submit = []
except IndexError:
    pass
except ValueError:
    pass
except TypeError:
    print('TypeError')
    sys.exit()
    #  train_latest_id_list = np.load(f'../input/card_id_train_first_active_2017{sys.argv[2]}.npy')
    #  test_latest_id_list = np.load(f'../input/card_id_test_first_active_2017{sys.argv[2]}.npy')
    #  train = train.loc[train[key].isin(train_latest_id_list), :].reset_index(drop=True)
    #  test = test.loc[test[key].isin(test_latest_id_list), :].reset_index(drop=True)
    #  submit = []
#========================================================================

if 'first_active_month' in list(train.columns):
    train.drop('first_active_month', axis=1, inplace=True)
    test.drop('first_active_month', axis=1, inplace=True)
y = train[target].values

# Target Check
#  y = np.round(y, 0)
#  print(pd.Series(y).value_counts())
#  sys.exit()

#========================================================================
# FM STACK
#  try:
#      if int(sys.argv[6])>0:
#          fm_feat = utils.read_pkl_gzip('../stack/0112_150_stack_keras_lr0_117feats_1seed_128.0batch_OUT_CV0-73219_feat_no_amount_only_ohe_first_month_category123_feature123_encode.gz')['prediction'].values
#          train['fm_keras'] = fm_feat[:len(train)]
#          test['fm_keras'] = fm_feat[len(train):]

#          fm_feat = utils.read_pkl_gzip('../stack/0112_234_stack_keras_lr0_72feats_1seed_128.0batch_OUT_CV0-688061879169805_LB.gz')['prediction'].values
#          train['fm_keras_2'] = fm_feat[:len(train)]
#          test['fm_keras_2'] = fm_feat[len(train):]
#  except IndexError:
#      pass
#========================================================================


if out_part=='part':
    # Exclude Difficult Outlier
    #  clf_result = utils.read_pkl_gzip('../stack/0111_145_outlier_classify_9seed_lgb_binary_CV0-9045939277654236_188features.gz')[[key, 'prediction']]
    clf_result = utils.read_pkl_gzip('../stack/0112_155_outlier_classify_9seed_lgb_binary_CV0-9047260065151934_200features.gz')[[key, 'pred_mean']]
    train = train.merge(clf_result, how='inner', on=key)
    #  tmp1 = train[train.prediction>0.01]
    #  tmp2 = train[train.prediction<0.01][train.target>-30]
    tmp1 = train[train.pred_mean>0.01]
    tmp2 = train[train.pred_mean<0.01][train.target>-30]
    train = pd.concat([tmp1, tmp2], axis=0, ignore_index=True)
    del tmp1, tmp2
    gc.collect()
    #  train.drop('prediction', axis=1, inplace=True)
    train.drop('pred_mean', axis=1, inplace=True)
elif out_part=='all':
    #  Exclude Outlier
    train = train[train.target>-30]

#========================================================================

#========================================================================
# LGBM Setting
try:
    argv3 = int(sys.argv[3])
    seed_list = np.arange(argv3)
    if argv3<=10:
        seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005, 2019][:argv3]
except IndexError:
    seed_list = [1208]
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

#  train, test, drop_list = LGBM.data_check(train=train, test=test, target=target)
train, test, drop_list = LGBM.data_check(train=train, test=test, target=target, encode='label')
if len(drop_list):
    train.drop(drop_list, axis=1, inplace=True)
    test.drop(drop_list, axis=1, inplace=True)

from sklearn.model_selection import StratifiedKFold

# seed_avg
seed_pred = np.zeros(len(test))
cv_list = []
iter_list = []
model_list = []
for i, seed in enumerate(seed_list):

    LGBM = lgb_ex(logger=logger, metric=metric, model_type=model_type, ignore_list=ignore_list)
    LGBM.seed = seed

    #  if i>=5:
    #      params['num_leaves'] = 48
    #      params['subsample'] = 0.8757099996397999
    #      params['colsample_bytree'] = 0.7401342964627846
    #      params['min_child_samples'] = 61

    if len(train)>150000 and pl_length==0:
        train['outliers'] = train[target].map(lambda x: 1 if x<-3 else 0)
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        kfold = folds.split(train,train['outliers'].values)
        train.drop('outliers', axis=1, inplace=True)

    elif pl_length > 0 or int(sys.argv[2][:2]) != 0:
        # Psued Labelingだと、outlierと予測された値は-7 ~ -18あたりになりそう?
        train['outliers'] = train[target].map(lambda x: 1 if x<=-3 else 0)
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        kfold = folds.split(train,train['outliers'].values)
        train.drop('outliers', axis=1, inplace=True)

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

    if pl_length>0:
        LGBM.result_stack = LGBM.result_stack.reset_index(drop=True).iloc[:pl_length,:]

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

    try:
        # first month modelで全体に対する予測値を出してstackする特徴を保存する場合
        sys.argv[4]
        model_list += LGBM.fold_model_list
    except IndexError:
        pass


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
# First Month Pred For Stack
# expert modelを作成した際、全体に対する予測値を出してStackingできるようにする
try:
    sys.argv[4]
    use_cols = LGBM.use_cols

    if sys.argv[2][-2:]=='pl':
        if int(sys.argv[2][:2])<12:
            df_pred = base.merge(df_pred.drop(target, axis=1), how='inner', on=key)
            pred = df_pred['pred_mean'].values
        else:
            pred = df_pred['pred_mean'].values
        y_train = y

    else:
        # Reload Dataset
        base = utils.read_df_pkl('../input/base_first0*')
        base_train = base[~base[target].isnull()].reset_index(drop=True)
        base_test = base[base[target].isnull()].reset_index(drop=True)
        feature_list = utils.parallel_load_data(path_list=win_path_list)
        df_feat = pd.concat(feature_list, axis=1)
        train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
        test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
        train_test = pd.concat([train, test], axis=0, ignore_index=True)[use_cols]
        y_train = train[target].values

        pred = np.zeros(len(train_test))
        for model in model_list:
            pred += model.predict(train_test)
        pred /= len(model_list)

    y_pred = pred[:len(y_train)]
    score = np.sqrt(mean_squared_error(np.where(y_train!=y_train, 0, y_train), y_pred))

    utils.to_pkl_gzip(obj=pred, path=f"../model/2017{sys.argv[2][:2]}/stack/{start_time[4:13]}_elo_first_month2017{sys.argv[2]}_{len(seed_list)}seed_lr{str(learning_rate).replace('.', '-')}_round{num_boost_round}_CV{str(score)[:6].replace('.', '-')}")
    #  pd.Series(LGBM.use_cols, name='use_cols').to_csv( f'../model/2017{sys.argv[2][:2]}/stack/{start_time[4:8]}_elo_first_month2017{sys.argv[2]}_fold_model_use_cols.csv',  index=False)
except IndexError:
    pass
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

#========================================================================
# Part of card_id Score
try:
    if pl_length>0 or sys.argv[2]=='all' or len(train)>150000:
        for i in range(201701, 201713, 1):
            train_latest_id_list = np.load(f'../input/card_id_train_first_active_{i}.npy')

            part_train = df_pred.loc[df_pred[key].isin(train_latest_id_list), :]
            y_train = part_train[target].values
            if 'pred_mean' in list(part_train.columns):
                y_pred = part_train['pred_mean'].values
            else:
                y_pred = part_train['prediction'].values
            part_score = np.sqrt(mean_squared_error(y_train, y_pred))

            logger.info(f'''
            #========================================================================
            # First Month {i} of Score: {part_score} | N: {len(train_latest_id_list)}
            #========================================================================''')
except ValueError:
    pass
except TypeError:
    pass
#========================================================================


try:
    sys.argv[4]
    out_score = 0
except IndexError:
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
        path_list = glob.glob('../log_submit/01*CV*LB*.csv')
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
