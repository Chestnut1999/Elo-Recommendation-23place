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
path_list = glob.glob('../model/201712/stack/*.gz')

#========================================================================
# First Month Group Score
for ratio_1, ratio_2 in zip(np.arange(0.1, 1.0, 0.1), np.arange(0.9, 0.0, -0.1)):

    base['prediction'] = 0
    for path in path_list:
        if not(path.count('stack')) or not(path.count('')):continue
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        pred = utils.read_pkl_gzip(path)
        if path.count('201712_all'):
            base['prediction'] += pred * ratio_1
        elif path.count('201712_org'):
            base['prediction'] += pred * ratio_2

    #========================================================================
    # Part of card_id Score
    for i in range(201701, 201713, 1):
        train_latest_id_list = np.load(f'../input/card_id_train_first_active_{i}.npy')

        df_part = base.loc[base[key].isin(train_latest_id_list), :]
        y_train = df_part[target].values
        y_pred = df_part['prediction'].values
        part_score = np.sqrt(mean_squared_error(y_train, y_pred))

        logger.info(f'''
        #========================================================================
        # First Month {i} of Score: {part_score} | N: {len(train_latest_id_list)}
        # Ratio1: {ratio_1} | Ratio2: {ratio_2}
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
