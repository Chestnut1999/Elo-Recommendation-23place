fold_seed = 328
outlier_thres = -3
num_threads = 32
#  num_threads = 36
import sys
import pandas as pd

out_part = 'clf_out'
model_no = int(sys.argv[2])
valid_type = sys.argv[3]

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
col_term = 'hist_regist_term'
no_flg = 'no_out_flg'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term', col_term, no_flg, 'clf_pred']

stack_name = out_part

model_type='lgb'
try:
    learning_rate = float(sys.argv[1])
except ValueError:
    learning_rate = 0.01
early_stopping_rounds = 200
num_boost_round = 15000

import numpy as np
import datetime
import glob
import re
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
num_leaves = 57
#  num_leaves = 59
#  num_leaves = 61
#  num_leaves = 68
num_leaves = 70
#  num_leaves = 71
params['num_leaves'] = num_leaves
params['num_threads'] = num_threads

try:
    num_leaves = int(sys.argv[4])
    params['num_leaves'] = num_leaves
except IndexError:
    pass

if num_leaves==71:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.3180226
    params['min_child_samples'] = 31
    params['lambda_l2'] = 14
elif num_leaves==70:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.325582
    params['min_child_samples'] = 30
    params['lambda_l2'] = 7
elif num_leaves>65:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2755158
    params['min_child_samples'] = 37
    params['lambda_l2'] = 7

elif num_leaves>60:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2792
    params['min_child_samples'] = 59
    params['lambda_l2'] = 2

elif num_leaves==57:
    params['subsample'] = 0.9
    params['colsample_bytree'] = 0.2513
    params['min_child_samples'] = 32
    params['lambda_l2'] = 9

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
colsample_bytree = params['colsample_bytree']
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Data Load

win_path = f'../features/4_winner/*.gz'
#  win_path = f'../features/1_first_valid/*.gz'
model_path_list = [f'../model/LB3670_70leaves_colsam0322/*.gz', '../model/E2_lift_set/*.gz', '../model/E3_PCA_set/*.gz', '../model/E4_mix_set/*.gz']
model_path = model_path_list[model_no]
#  tmp_path_list = glob.glob(f'../features/5_tmp/*.gz') + glob.glob(f'../features/0_exp/*.gz')
#  tmp_path_list = glob.glob(f'../features/5_tmp/*.gz')
#  win_path_list = glob.glob(model_path) + glob.glob(win_path) + glob.glob(f'../features/5_tmp/*.gz')
win_path_list = glob.glob(model_path)
#  win_path_list = glob.glob(win_path) + tmp_path_list

base = utils.read_pkl_gzip('../input/base_no_out_clf.gz')[[key, target, col_term, 'first_active_month', no_flg, 'clf_pred']]
#  base = utils.read_df_pkl('../input/base_term*')[[key, target, col_term, 'first_active_month']]
base[col_term] = base[col_term].map(lambda x:
                                          24 if 19<=x else
                                          18 if 16<=x and x<=18 else
                                          15 if 13<=x and x<=15 else
                                          12 if 9<=x and x<=12  else
                                          8 if 6<=x and x<=8    else
                                          5 if x==5 else
                                          4
                                         )

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)

feature_list = utils.parallel_load_data(path_list=win_path_list)

df_feat = pd.concat(feature_list, axis=1)

train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)

self_predict = []
self_predict = train.copy()
# 閾値以上のoutlierのidを除いてTraining
upper_thres = float(sys.argv[5])
lower_thres = float(sys.argv[6])
tmp1 = train[train['clf_pred']>upper_thres]
tmp2 = train[train['clf_pred']<lower_thres]
tmp1 = tmp1[tmp1[target]<-30]
tmp2 = tmp2[tmp2[target]<-30]
rm_clf_out_id_list = list(tmp1[key].values) + list(tmp2[key].values)
train = train.loc[~train[key].isin(rm_clf_out_id_list), :]
train.drop('clf_pred', axis=1, inplace=True)

y = train[target].values

#========================================================================

#========================================================================
# LGBM Setting
try:
    argv3 = int(sys.argv[7])
    seed_list = np.arange(argv3)
    if argv3<=10:
        seed_list = [1208, 605, 1212, 1222, 405, 1128, 1012, 328, 2005, 2019][:argv3]
        seed_list = [328, 605, 1212, 1222, 405, 1128, 1012, 1208, 2005, 2019][:argv3]
except IndexError:
    seed_list = [1208]
    seed_list = [328]
metric = 'rmse'
#  metric = 'mse'
params['metric'] = metric
fold=6
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
    if len(self_predict)>0:
        self_predict.drop(drop_list, axis=1, inplace=True)

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

    #  if i>=5:
    #      params['num_leaves'] = 48
    #      params['subsample'] = 0.8757099996397999
    #      params['colsample_bytree'] = 0.7401342964627846
    #      params['min_child_samples'] = 61

    #========================================================================
    # Validation Setting vvv
    # Validation Set はFitさせたいFirst month のグループに絞る
    # 1. マイナスでOutlierの閾値を切って、それらの分布が揃う様にKFoldを作る
    kfold_path = f'../input/kfold_{valid_type}_{out_part}_fold{fold}_seed{fold_seed}.gz'
    if os.path.exists(kfold_path) and out_part!='clf_out':
        kfold = utils.read_pkl_gzip(kfold_path)

    # 2. プラスマイナスでOutlierの閾値を切って、プラス、マイナス別に分布が揃う様にKFoldを作る
    elif valid_type=='pmo':
        plus  = train[train[target] >= 0]
        tmp_minus = train[train[target] <  0]
        minus = tmp_minus[tmp_minus[target] >  -30]
        out = tmp_minus[tmp_minus[target] <  -30]

        plus['outliers'] = plus[target].map(lambda x: 1 if x>=outlier_thres*-1 else 0)
        minus['outliers'] = minus[target].map(lambda x: 1 if x<=outlier_thres else 0)
        out['outliers'] = out[target].map(lambda x: 1 if x<=outlier_thres else 0)

        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=fold_seed)
        kfold_plus = folds.split(plus, plus['outliers'].values)
        kfold_minus = folds.split(minus, minus['outliers'].values)
        kfold_out = folds.split(out, out['outliers'].values)

        trn_idx_list = []
        val_idx_list = []
        for (p_trn_idx, p_val_idx), (m_trn_idx, m_val_idx), (o_trn_idx, o_val_idx) in zip(kfold_plus, kfold_minus, kfold_out):

            def get_ids(df, idx):
                ids = list(df.iloc[idx, :][key].values)
                return ids

            trn_ids = get_ids(plus, p_trn_idx) + get_ids(minus, m_trn_idx) + get_ids(out, o_trn_idx)
            val_ids = get_ids(plus, p_val_idx) + get_ids(minus, m_val_idx) + get_ids(out, o_val_idx)

            # idをindexの番号にする
            #  trn_ids = list(train[train[key].isin(trn_ids)].index)
            #  val_ids = list(train[train[key].isin(val_ids)].index)

            trn_idx_list.append(trn_ids)
            val_idx_list.append(val_ids)
        kfold = [trn_idx_list, val_idx_list]

    elif valid_type=='pm':
        plus  = train[train[target] >= 0]
        minus = train[train[target] <  0]
        #  minus = tmp_minus[tmp_minus[target] >  -30]

        plus['outliers'] = plus[target].map(lambda x: 1 if x>=outlier_thres*-1 else 0)
        minus['outliers'] = minus[target].map(lambda x: 1 if x<=outlier_thres else 0)

        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=fold_seed)
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
            #  trn_ids = list(train[train[key].isin(trn_ids)].index)
            #  val_ids = list(train[train[key].isin(val_ids)].index)

            trn_idx_list.append(trn_ids)
            val_idx_list.append(val_ids)
        # card_id ver
        kfold = [trn_idx_list, val_idx_list]

    elif valid_type=='out':

        train['outliers'] = train[target].map(lambda x: 1 if x<-30 else 0)
        folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=fold_seed)
        kfold = list(folds.split(train,train['outliers'].values))
        train.drop('outliers', axis=1, inplace=True)

        # card_id listにする
        trn_list = []
        val_list = []
        for trn, val in kfold:
            trn_ids = train.iloc[trn][key].values
            val_ids = train.iloc[val][key].values
            trn_list.append(trn_ids)
            val_list.append(val_ids)
        kfold = [trn_list, val_list]

    elif valid_type=='term':
        outlier_thres = -3

        term_list = list(train[col_term].value_counts().index)

        trn_idx_list = []
        val_idx_list = []
        train_dict = {}
        valid_dict = {}
        for term in term_list:
            df  = train[train[col_term] == term]

            plus  = df[df[target] >= 0]
            tmp_minus = df[df[target] <  0]
            minus = tmp_minus[tmp_minus[target] >  -30]
            out = tmp_minus[tmp_minus[target] <  -30]

            plus['outliers'] = plus[target].map(lambda x: 1 if x>=outlier_thres*-1 else 0)
            minus['outliers'] = minus[target].map(lambda x: 1 if x<=outlier_thres else 0)
            out['outliers'] = out[target].map(lambda x: 1 if x<=outlier_thres else 0)

            folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=fold_seed)
            kfold_plus = folds.split(plus, plus['outliers'].values)
            kfold_minus = folds.split(minus, minus['outliers'].values)
            if len(out):
                kfold_out = folds.split(out, out['outliers'].values)
            else:
                kfold_out = zip(range(fold), range(fold))

            for fold_num, ((p_trn_idx, p_val_idx), (m_trn_idx, m_val_idx), (o_trn_idx, o_val_idx)) in enumerate(zip(kfold_plus, kfold_minus, kfold_out)):

                def get_ids(df, idx):
                    ids = list(df.iloc[idx, :][key].values)
                    return ids

                if len(out):
                    trn_ids = get_ids(plus, p_trn_idx) + get_ids(minus, m_trn_idx) + get_ids(out, o_trn_idx)
                    val_ids = get_ids(plus, p_val_idx) + get_ids(minus, m_val_idx) + get_ids(out, o_val_idx)
                else:
                    trn_ids = get_ids(plus, p_trn_idx) + get_ids(minus, m_trn_idx)
                    val_ids = get_ids(plus, p_val_idx) + get_ids(minus, m_val_idx)

                # idをindexの番号にする
                #  trn_ids = list(df[df[key].isin(trn_ids)].index)
                #  val_ids = list(df[df[key].isin(val_ids)].index)

                if fold_num not in train_dict:
                    train_dict[fold_num] = trn_ids
                    valid_dict[fold_num] = val_ids
                else:
                    train_dict[fold_num] += trn_ids
                    valid_dict[fold_num] += val_ids
            print(len(train_dict[fold_num]), len(valid_dict[fold_num]))
        #  kfold = list(zip(train_dict.values(), valid_dict.values()))
        kfold = [train_dict.values(), valid_dict.values()]

    elif valid_type=='ods':
        train['rounded_target'] = train['target'].round(0)
        train = train.sort_values('rounded_target').reset_index(drop=True)
        vc = train['rounded_target'].value_counts()
        vc = dict(sorted(vc.items()))
        df = pd.DataFrame()
        train['indexcol'],idx = 0,1
        for k,v in vc.items():
            step = train.shape[0]/v
            indent = train.shape[0]/(v+1)
            df2 = train[train['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
            for j in range(0, v):
                df2.at[j, 'indexcol'] = indent + j*step + 0.000001*idx
            df = pd.concat([df2,df])
            idx+=1
        train = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
        del train['indexcol'], train['rounded_target']

        folds = KFold(n_splits=6, shuffle=False, random_state=fold_seed)
        kfold = list(folds.split(train, train[target].values))

        # card_id listにする
        trn_list = []
        val_list = []
        for trn, val in kfold:
            trn_ids = train.iloc[trn][key].values
            val_ids = train.iloc[val][key].values
            trn_list.append(trn_ids)
            val_list.append(val_ids)
        kfold = [trn_list, val_list]

    elif valid_type=='ods_term':

        outlier_thres = -3
        term4  = train[train[col_term] == 4]
        term5  = train[train[col_term] == 5]
        term6  = train[train[col_term] == 6]
        term9  = train[train[col_term] == 9]
        term15  = train[train[col_term] == 15]
        term18  = train[train[col_term] == 18]
        term24  = train[train[col_term] == 24]

        df_list = [
        term4
        ,term5
        ,term6
        ,term9
        ,term15
        ,term18
        ,term24
        ]


        fold_type = 'self'
        trn_idx_list = []
        val_idx_list = []
        train_dict = {}
        valid_dict = {}

        for df_term in df_list:

            df_term['rounded_target'] = df_term['target'].round(0)
            df_term = df_term.sort_values('rounded_target').reset_index(drop=True)
            vc = df_term['rounded_target'].value_counts()
            vc = dict(sorted(vc.items()))
            df = pd.DataFrame()
            df_term['indexcol'],idx = 0,1
            for k,v in vc.items():
                step = df_term.shape[0]/v
                indent = df_term.shape[0]/(v+1)
                df2 = df_term[df_term['rounded_target'] == k].sample(v, random_state=fold_seed).reset_index(drop=True)
                for j in range(0, v):
                    df2.at[j, 'indexcol'] = indent + j*step + 0.000001*idx
                df = pd.concat([df2,df])
                idx+=1
            df_term = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
            del df_term['indexcol'], df_term['rounded_target']
            folds = KFold(n_splits=fold, shuffle=False, random_state=fold_seed)
            kfold = folds.split(df_term, df_term[target].values)

            for fold_num, (p_trn_idx, p_val_idx) in enumerate(kfold):

                def get_ids(df, idx):
                    ids = list(df.iloc[idx, :][key].values)
                    return ids

                trn_ids = get_ids(df_term, p_trn_idx)
                val_ids = get_ids(df_term, p_val_idx)

                # idをindexの番号にする
                #  trn_ids = list(train[train[key].isin(trn_ids)].index)
                #  val_ids = list(train[train[key].isin(val_ids)].index)

                if fold_num not in train_dict:
                    train_dict[fold_num] = trn_ids
                    valid_dict[fold_num] = val_ids
                else:
                    train_dict[fold_num] += trn_ids
                    valid_dict[fold_num] += val_ids
            print(len(np.unique(train_dict[fold_num])), len(np.unique(valid_dict[fold_num])))
        kfold = list(zip(train_dict.values(), valid_dict.values()))


    # 3. Default KFold
    else:
        kfold = False
        fold_type = 'kfold'
    #========================================================================
    if not(os.path.exists(kfold_path)):
        utils.to_pkl_gzip(obj=kfold, path=kfold_path)

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
        ,self_predict=self_predict
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
# Result
cv_score = np.mean(cv_list)
iter_avg = np.int(np.mean(iter_list))
#========================================================================

logger.info(f'''
#========================================================================
# {len(seed_list)}SEED CV SCORE AVG: {cv_score}
#========================================================================''')

#========================================================================
# STACKING
if len(stack_name)>0:
    logger.info(f'result_stack shape: {df_pred.shape}')
    if len(seed_list)>1:
        pred_cols = [col for col in df_pred.columns if col.count('predict')]
        df_pred['pred_mean'] = df_pred[pred_cols].mean(axis=1)
        df_pred['pred_std'] = df_pred[pred_cols].std(axis=1)
# Save
out_score = 0
pred_col = [col for col in df_pred.columns if col.count('pred')][0]
utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_{stack_name}_{model_type}_out_part-{out_part}_valid-{valid_type}_foldseed{fold_seed}_ESET{model_no}_row{len(train)}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_colsample{colsample_bytree}_iter{iter_avg}_OUT0_CV{str(cv_score).replace('.', '-')}_LB", obj=df_pred[[key, pred_col]])
#========================================================================

#========================================================================
# Part of card_id Score
#  bench = pd.read_csv('../input/bench_LB3684_FAM_cv_score.csv')
bench = utils.read_pkl_gzip('../stack/0206_125_stack_lgb_lr0.01_235feats_10seed_70leaves_iter1164_OUT29.8269_CV3-6215750935280235_LB.gz')[[key, 'pred_mean']].rename(columns={'pred_mean':'bench_pred'})
df_pred = df_pred.merge(bench, on=key, how='inner')
part_score_list = []
part_N_list = []
fam_list = []
base_train['first_active_month'] = base_train['first_active_month'].map(lambda x: str(x)[:7])

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
    bench_pred = part_train['bench_pred'].values

    # RMSE
    part_score = np.sqrt(mean_squared_error(y_train, y_pred))
    bench_score = np.sqrt(mean_squared_error(y_train, bench_pred))
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

#========================================================================
# Term
part_score_list = []
part_N_list = []
fam_list = []
term_list = sorted(list(base_train['hist_regist_term'].value_counts().index))
for term in term_list:
    df_part = base_train[base_train['hist_regist_term']==term]
    if len(df_part)<1:
        continue
    part_id_list = df_part[key].values

    part_train = df_pred.loc[df_pred[key].isin(part_id_list), :]

    y_train = part_train[target].values
    if 'pred_mean' in list(part_train.columns):
        y_pred = part_train['pred_mean'].values
    else:
        y_pred = part_train['prediction'].values
    bench_pred = part_train['bench_pred'].values

    # RMSE
    part_score = np.sqrt(mean_squared_error(y_train, y_pred))
    bench_score = np.sqrt(mean_squared_error(y_train, bench_pred))
    part_score -= bench_score

    fam_list.append(term)
    part_score_list.append(part_score)
    part_N_list.append(len(part_id_list))

#  for i, part_score, N in zip(fam_list, part_score_list, part_N_list):
df = pd.DataFrame(np.asarray([fam_list, part_score_list, part_N_list]).T)
df.columns = ['TERM', 'CV', 'N']

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

# 不要なカラムを削除
drop_feim_cols = [col for col in cv_feim.columns if col.count('importance_') or col.count('rank_')]
cv_feim.drop(drop_feim_cols, axis=1, inplace=True)
drop_feim_cols = [col for col in cv_feim.columns if col.count('importance') and not(col.count('avg'))]
cv_feim.drop(drop_feim_cols, axis=1, inplace=True)
cv_feim.to_csv( f'../valid/{start_time[4:12]}_valid_{model_type}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv' , index=False)

#========================================================================
# Submission
test_pred = seed_pred / len(seed_list)
submit[target] = test_pred
submit_path = f'../submit/{start_time[4:12]}_submit_{model_type}_obj{objective}_lr{learning_rate}_{feature_num}feats_{len(seed_list)}seed_{num_leaves}leaves_iter{iter_avg}_OUT{str(out_score)[:7]}_CV{cv_score}_LB.csv'
submit.to_csv(submit_path, index=False)
#========================================================================

#========================================================================
# Corr
#  base = utils.read_df_pkl('../input/base_term*')[[key, target]].set_index(key)
#  ens_list = glob.glob('../ensemble/*.gz')

#  if 'pred_mean' in df_pred.columns:
#      base['this_pred'] = df_pred.set_index(key)['pred_mean']
#  else:
#      base['this_pred'] = df_pred.set_index(key)['prediction']

#  # 相関
#  for i, path in enumerate(ens_list):
#      ens_model = utils.read_pkl_gzip(path)
#      if 'pred_mean' in ens_model.columns:
#          pred_col = 'pred_mean'
#      else:
#          pred_col = 'prediction'
#      ens_model = ens_model[[key, pred_col]].set_index(key)
#      base['ens_pred'] = ens_model[pred_col]
#      cv_score = re.search(r'CV([^/.]*)_LB.gz', path).group(1)
#      corr = np.corrcoef(base['ens_pred'], base['this_pred'].values).min()
#      logger.info(f"WITH CV{cv_score[:6]} CORR: {corr}")
#========================================================================
