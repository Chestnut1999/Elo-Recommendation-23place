import sys
import pandas as pd

#========================================================================
# Args
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id']

import numpy as np
import datetime
import glob
import re
import os
from sklearn.metrics import mean_squared_error

HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func
logger=logger_func()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# Data Load
base = utils.read_df_pkl('../input/base_first*')
path_list = glob.glob('../ensemble/*.gz')
#  path = '../stack/0127_120_stack_lgb_lr0.01_349feats_1seed_31leaves_iter3915_OUT0_CV1-139620018388889_LB.gz'
path_1 = '../ensemble/0112_123_stack_lgb_lr0.01_200feats_10seed_iter1121_OUT30.2024_CV3-649256498211181_LB3.687.gz'
path_2 = '../ensemble/0112_084_stack_lgb_lr0.01_200feats_10seed_OUT30.2199_CV3-649046125233803_LB3.687.gz'

#========================================================================
# First Month Group Score
#  for ratio_1, ratio_2 in zip(np.arange(0.1, 1.0, 0.1), np.arange(0.9, 0.0, -0.1)):
base['prediction'] = 0
#  filename = re.search(r'/([^/.]*).gz', path.replace('.', '-')).group(1)
pred_1 = utils.read_pkl_gzip(path_1)
pred_2 = utils.read_pkl_gzip(path_2)

base.set_index('card_id', inplace=True)
pred_1.set_index('card_id', inplace=True)
base['pred_1'] = pred_1['prediction']
base['pred_2'] = pred_2['prediction']
base['prediction'] = (base['pred_1'] + base['pred_2']) / 2
base['prediction'] = base['pred_1']
#  base['prediction'] = base['pred_2']
base.reset_index(inplace=True)
base = base[~base[target].isnull()]

#========================================================================
# Part of card_id Score
part_score_list = []
part_N_list = []
fam_list = []
#  for i in range(201101, 201713, 1):
for i in range(201501, 201713, 1):
    fam = str(i)[:4] + '-' + str(i)[-2:]
    df_part = base[base['first_active_month']==fam]
    if len(df_part)<1:
        continue
    part_id_list = df_part[key].values

    part_train = base.loc[base[key].isin(part_id_list), :]
    y_train = part_train[target].values
    if 'pred_mean' in list(part_train.columns):
        y_pred = part_train['pred_mean'].values
    else:
        y_pred = part_train['prediction'].values

    y_pred = np.where(y_pred != y_pred, 0, y_pred)
    # RMSE
    part_score = np.sqrt(mean_squared_error(y_train, y_pred))

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
