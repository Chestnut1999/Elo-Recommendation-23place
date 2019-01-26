#========================================================================
# argv[1]: Dataset(all, past, future, only)
# argv[2]: First_Month, Feature_Set, Psuedo, 
# argv[3]: Seed Num
# argv[4]: Save Stack Pred
# argv[5]: Validation Setting(Plus, PlusMinus, Default)
#========================================================================
import sys
out_part = ['', 'part', 'all'][0]
# Dataset Grouping
dataset_type = sys.argv[1]
try:
    outlier_thres = int(dataset_type[-2:])*-1
except ValueError:
    outlier_thres = -30
fm_feat_pl = sys.argv[2]

import sys
import pandas as pd

#========================================================================
# Args
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id']

fname=''
#========================================================================


model_type='lgb'
learning_rate = 0.02
early_stopping_rounds = 150
num_boost_round = 75000
num_boost_round = 10000
#  learning_rate = 0.1
#  num_threads = -1
num_threads = 36

import numpy as np
import datetime
import glob
import gc
import os
from sklearn.metrics import mean_squared_error
HOME = os.path.expanduser('~')

sys.path.append(f'{HOME}/kaggle/data_analysis/model')
sys.path.append(f'{HOME}/kaggle/data_analysis')
from model.lightgbm_ex import lightgbm_ex as lgb_ex

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

from params_lgbm import params_elo
params = params_elo()[1]
params['learning_rate'] = learning_rate

# Best outlier fit LB3.690
#  num_leaves = 4
#  num_leaves = 16
num_leaves = 31
#  num_leaves = 48
params['num_leaves'] = num_leaves
params['num_threads'] = num_threads
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
    if int(fm_feat_pl)>0:
        win_path = f'../model/2017{fm_feat_pl}/4_winner/*.gz'
        tmp_path_list = glob.glob('../model/2017{fm_feat_pl}/5_tmp/*.gz')
    # features配下のwinnerを使う時
    else:
        win_path = f'../features/4_winner/*.gz'
        tmp_path_list = glob.glob('../features/5_tmp/*.gz')
        # features配下のセットを使いつつ、first_monthを絞りたい場合
        if int(fm_feat_pl)<0:
            if int(fm_feat_pl)<10:
                fm_feat_pl = '0' + str(int(fm_feat_pl)*-1)
            else:
                fm_feat_pl = str(int(fm_feat_pl)*-1)
except ValueError:
    # ALLのwinnerを使う時
    if fm_feat_pl=='all':
        win_path = f'../model/all/4_winner/*.gz'
        tmp_path_list = glob.glob(f'../model/all/5_tmp/*.gz')
    else:
        # スードラベリングをするとき
        if fm_feat_pl[-2:]=='pl':
            win_path = f'../model/2017{fm_feat_pl[:2]}/{fm_feat_pl[2:-2]}/*.gz'
            tmp_path_list = glob.glob(f'../model/2017{fm_feat_pl[:2]}/5_tmp/*.gz')
        # スードラベリングをしないとき
        else:
            win_path = f'../model/2017{fm_feat_pl[:2]}/{fm_feat_pl[2:]}/*.gz'
            #  tmp_path_list = glob.glob(f'../model/2017{fm_feat_pl[:2]}/5_tmp/*.gz')
            tmp_path_list = glob.glob(f'../features/exp/*.gz')

## ddd
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


train_latest_id_list = np.load(f'../input/card_id_train_first_active_2017{fm_feat_pl[:2]}.npy')
test_latest_id_list = np.load(f'../input/card_id_test_first_active_2017{fm_feat_pl[:2]}.npy')

#========================================================================
# card_id list by first active month
try:
    if int(fm_feat_pl[:2])>0:
        first_month = f'2017-{fm_feat_pl[:2]}'

        if fm_feat_pl[-2:]=='pl':
            pred_path = glob.glob(f'../model/2017{fm_feat_pl[:2]}/stack/*org0_*')[0]
            pred_col = 'pred'
            pred_feat = utils.read_pkl_gzip(pred_path)
            train[pred_col] = pred_feat[:len(train)]
            train.loc[~train[key].isin(train_latest_id_list), target] = train.loc[~train[key].isin(train_latest_id_list), pred_col]

            tmp_test = test.copy()
            tmp_test[target] = pred_feat[len(train):]

            # first_active_monthが201712より前の場合、学習データセットから未来のfirst_active_monthを除外する
            if int(fm_feat_pl[:2])<12:
                base = base[base['first_active_month'] <= f'2017-{fm_feat_pl[:2]}']
                train = train.merge(base[key].to_frame(), how='inner', on=key)
                test = test.merge(base[key].to_frame(), how='inner', on=key)
                tmp_test = tmp_test.merge(base[key].to_frame(), how='inner', on=key)


            train = pd.concat([train, tmp_test], axis=0, ignore_index=True).drop(pred_col, axis=1)
            pl_length = len(train)
            del tmp_test
            gc.collect()
        else:
            # ここでデータセットを好きなグループに分ける
            # 1. All
            # 2. =< First Month
            # 3. >= First Month
            # 4. == First Month
            if dataset_type.count('all'):
                pass
            elif dataset_type.count('past_'):
                base = base[base['first_active_month'] <= f'2017-{fm_feat_pl[:2]}']
                train = train.merge(base[key].to_frame(), how='inner', on=key)
                test = test.merge(base[key].to_frame(), how='inner', on=key)
            elif dataset_type.count('past3'):
                base = base[base['first_active_month'] <= f'2017-{fm_feat_pl[:2]}']
                past3 = int(fm_feat_pl[:2]) - int(sys.argv[6])
                fm_feat_pl = fm_feat_pl.replace('past3', f'past{sys.argv[6]}')
                if past3<10:
                    past3 = f'0{past3}'
                base = base[base['first_active_month'] >  f'2017-{past3}']
                train = train.merge(base[key].to_frame(), how='inner', on=key)
                test = test.merge(base[key].to_frame(), how='inner', on=key)
            elif dataset_type.count('future'):
                base = base[base['first_active_month'] >= f'2017-{fm_feat_pl[:2]}']
                train = train.merge(base[key].to_frame(), how='inner', on=key)
                test = test.merge(base[key].to_frame(), how='inner', on=key)
            elif dataset_type.count('only'):
                train = train.loc[train[key].isin(train_latest_id_list), :].reset_index(drop=True)
                test = test.loc[test[key].isin(test_latest_id_list), :].reset_index(drop=True)
except IndexError:
    pass
except ValueError:
    pass
except TypeError:
    print('TypeError')
    sys.exit()

if dataset_type.count('dist'):
    max_train = train.loc[train[key].isin(train_latest_id_list), target].max()
    min_train = train.loc[train[key].isin(train_latest_id_list), target].min()
    train = train[train[target]<=max_train]
    train = train[train[target]>=min_train]


#========================================================================
# Loyalty PreProcessing
# 1. どこまでOutlierをデータセットに含めるか
# 2. Loyaltyのどこまでの範囲をデータセットに含めるか

# 1. Classifierの確率を使い、Outlierの確率が高いグループのみ残す
if out_part=='part':
    # Exclude Difficult Outlier
    #  clf_result = utils.read_pkl_gzip('../stack/0111_145_outlier_classify_9seed_lgb_binary_CV0-9045939277654236_188features.gz')[[key, 'prediction']]
    clf_result = utils.read_pkl_gzip('../stack/0112_155_outlier_classify_9seed_lgb_binary_CV0-9047260065151934_200features.gz')[[key, 'pred_mean']]
    train = train.merge(clf_result, how='inner', on=key)
    tmp1 = train[train.pred_mean>0.01]
    tmp2 = train[train.pred_mean<0.01][train.target>-30]
    train = pd.concat([tmp1, tmp2], axis=0, ignore_index=True)
    del tmp1, tmp2
    gc.collect()
    #  train.drop('prediction', axis=1, inplace=True)
    train.drop('pred_mean', axis=1, inplace=True)

# 2. Outlierを閾値で切って全て除外する
elif out_part=='all':
    #  Exclude Outlier
    train = train[train.target > outlier_thres]

#========================================================================

if 'first_active_month' in list(train.columns):
    train.drop('first_active_month', axis=1, inplace=True)
    test.drop('first_active_month', axis=1, inplace=True)
train.reset_index(drop=True, inplace=True)
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



#========================================================================
# LGBM Setting
try:
    seed_num = int(sys.argv[3])
    seed_list = np.arange(seed_num)
    if seed_num<=10:
        seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005, 2019][:seed_num]
except IndexError:
    seed_list = [1208]
metric = 'rmse'
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

    #========================================================================
    # Validation Setting vvv
    #  fm_idx_list = list(base_train[base_train['first_active_month'] == f'2017-{fm_feat_pl[:2]}'].index)
    # Validation Set はFitさせたいFirst month のグループに絞る
    # 1. マイナスでOutlierの閾値を切って、それらの分布が揃う様にKFoldを作る
    if sys.argv[4]=='minus':
        train['outliers'] = train[target].map(lambda x: 1 if x < outlier_thres else 0)
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        kfold = folds.split(train,train['outliers'].values)
        train.drop('outliers', axis=1, inplace=True)

    # 2. プラスマイナスでOutlierの閾値を切って、プラス、マイナス別に分布が揃う様にKFoldを作る
    elif sys.argv[4]=='pm':
        plus  = train[train[target] >= 0]
        minus = train[train[target] <  0]

        plus['outliers'] = plus[target].map(lambda x: 1 if x>=outlier_thres*-1 else 0)
        minus['outliers'] = minus[target].map(lambda x: 1 if x<=outlier_thres else 0)
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        kfold_plus = folds.split(plus, plus['outliers'].values)
        kfold_minus = folds.split(minus, minus['outliers'].values)

        trn_idx_list = []
        val_idx_list = []
        for (p_trn_idx, p_val_idx), (m_trn_idx, m_val_idx) in zip(kfold_plus, kfold_minus):

            def get_ids(df, idx):
                ids = list(df.iloc[idx, :][key].values)
                return ids

            trn_ids = get_ids(plus, p_trn_idx) + get_ids(minus, m_trn_idx)
            val_ids = get_ids(plus, p_val_idx) + get_ids(minus, m_val_idx)

            # idをindexの番号にする
            trn_ids = list(train[train[key].isin(trn_ids)].index)
            val_ids = list(train[train[key].isin(val_ids)].index)

            trn_idx_list.append(trn_ids)
            val_idx_list.append(val_ids)
        kfold = zip(trn_idx_list, val_idx_list)


    elif sys.argv[4]=='fmpm':

        train.reset_index(drop=True, inplace=True)
        fm_train = train[train[key].isin(train_latest_id_list)].reset_index(drop=True)
        plus  = fm_train[fm_train[target] >= 0]
        minus = fm_train[fm_train[target] <  0]

        plus['outliers'] = plus[target].map(lambda x: 1 if x>=outlier_thres*-1 else 0)
        minus['outliers'] = minus[target].map(lambda x: 1 if x<=outlier_thres else 0)

        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        kfold_plus = folds.split(plus, plus['outliers'].values)
        kfold_minus = folds.split(minus, minus['outliers'].values)

        trn_idx_list = []
        val_idx_list = []
        for (p_trn_idx, p_val_idx), (m_trn_idx, m_val_idx) in zip(kfold_plus, kfold_minus):

            def get_ids(df, idx):
                ids = list(df.iloc[idx, :][key].values)
                return ids

            val_ids = get_ids(plus, p_val_idx) + get_ids(minus, m_val_idx)
            trn_ids = list(set(list(train[key].values)) - set(val_ids))

            # idをindexの番号にする
            trn_ids = list(train[train[key].isin(trn_ids)].index)
            val_ids = list(train[train[key].isin(val_ids)].index)

            trn_idx_list.append(trn_ids)
            val_idx_list.append(val_ids)

        kfold = zip(trn_idx_list, val_idx_list)

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

    if pl_length>0:
        LGBM.result_stack = LGBM.result_stack.reset_index(drop=True).iloc[:pl_length, :]

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
        sys.argv[5]
        model_list += LGBM.fold_model_list
    except IndexError:
        pass


#========================================================================
# STACKING
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
    stack_name = sys.argv[5]
    use_cols = LGBM.use_cols

    if fm_feat_pl[-2:]=='pl':
        if int(fm_feat_pl[:2])<12:
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

        # concatはindexで結合されるので注意
        train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1).reset_index()
        test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
        train_test = pd.concat([train, test], axis=0, ignore_index=True)[use_cols]

        # First_Month CV
        fm_idx = list(train[train[key].isin(train_latest_id_list)].index)
        y_train = train.iloc[fm_idx][target].values
        pred = np.zeros(len(train_test))
        for model in model_list:
            pred += model.predict(train_test)
        pred /= len(model_list)

    y_pred = pred[fm_idx]
    score = np.sqrt(mean_squared_error(np.where(y_train!=y_train, 0, y_train), y_pred))

    utils.to_pkl_gzip(obj=pred, path=f"../model/2017{fm_feat_pl[:2]}/stack/{start_time[4:13]}_elo_first_month2017{fm_feat_pl[:2]}_{fm_feat_pl[2:]}_{dataset_type}_{stack_name}_{len(seed_list)}seed_lr{str(learning_rate).replace('.', '-')}_round{num_boost_round}_CV{str(score)[:6].replace('.', '-')}")
    #  pd.Series(LGBM.use_cols, name='use_cols').to_csv( f'../model/2017{fm_feat_pl[:2]}/stack/{start_time[4:8]}_elo_first_month2017{fm_feat_pl}_fold_model_use_cols.csv',  index=False)
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
    if pl_length>0 or fm_feat_pl=='all' or len(train)>150000:
        #  for i in range(201701, 201713, 1):
        for i in range(201712, 201713, 1):
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
            # First Month {i} of Score: {part_score} | N: {len(part_train)}
            #========================================================================''')
except ValueError:
    pass
except TypeError:
    pass
utils.to_pkl_gzip(obj=df_pred, path='../stack/num_check')
#========================================================================


#  try:
out_score = 0
#  except IndexError:
#  if len(train)>150000:
#      if len(train[train[target] < outlier_thres])>0:
#          # outlierに対するスコアを出す
#          train.reset_index(inplace=True)
#          out_ids = train.loc[train.target < outlier_thres, key].values
#          out_val = train.loc[train.target < outlier_thres, target].values
#          if len(seed_list)==1:
#              out_pred = df_pred[df_pred[key].isin(out_ids)]['prediction'].values
#          else:
#              out_pred = df_pred[df_pred[key].isin(out_ids)]['pred_mean'].values
#          out_score = np.sqrt(mean_squared_error(out_val, out_pred))
#      else:
#          out_score = 0
#  else:
#      out_score = 0

# Save
try:
    if int(fm_feat_pl)==0:
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
cv_feim.to_csv( f'../valid/{start_time[4:12]}_valid_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_{dataset_type}_{fm_feat_pl}_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv' , index=False)


#========================================================================
# CV INFO
#  try:
#      if dataset_type != 'only':

#          import re
#          path_list = glob.glob('../log_submit/01*CV*LB*.csv')

#          tmp_list = []
#          path_list = list(set(path_list))
#          for path in path_list:
#              tmp = pd.read_csv(path)
#              tmp_path = path.replace(".", '-')
#              cv = re.search(r'CV([^/.]*)_LB', tmp_path).group(1).replace('-', '.')
#              lb = re.search(r'LB([^/.]*).csv', tmp_path).group(1).replace('-', '.')
#              tmp.rename(columns={'target':f"CV{cv[:9]}_LB{lb}"}, inplace=True)
#              tmp.set_index('card_id', inplace=True)
#              tmp_list.append(tmp.copy())

#          if len(tmp_list)>0:
#              df = pd.concat(tmp_list, axis=1)
#              df_corr = df.corr(method='pearson')

#              logger.info(f'''
#  #========================================================================
#  # OUTLIER FIT SCORE: {out_score}
#  # SUBMIT CORRELATION:
#  {df_corr[f'CV{str(cv_score)[:9]}_LB'].sort_values()}
#  #========================================================================''')
#  except ValueError:
#      pass
#  except TypeError:
#      pass
