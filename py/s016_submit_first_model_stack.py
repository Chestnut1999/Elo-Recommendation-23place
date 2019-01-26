import sys
import pandas as pd
import numpy as np
import datetime
import glob
import gc
import os
HOME = os.path.expanduser('~')

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping
from utils import logger_func

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'column_0']

win_path = f'../features/4_winner/*.gz'
#  win_path = f'../model/old_201712/*.gz'
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load
base = utils.read_df_pkl('../input/base_first*')
fm201712_all = utils.read_pkl_gzip('../model/201712/stack/0126_0933_elo_first_month201712_all_dist_all_03_stack_1seed_lr0-02_round75000_CV3-6547.gz')
fm201712_org = utils.read_pkl_gzip('../model/201712/stack/0126_0933_elo_first_month201712_org0_dist_all_03_stack_1seed_lr0-02_round75000_CV3-7252.gz')
base['prediction'] = fm201712_all*0.5 + fm201712_org*0.5
#========================================================================

submit = pd.read_csv('../log_submit/0117_081_elo_5blender_LB3.686.csv')

# First Month id list
fm1712_id = list(np.load('../input/card_id_test_first_active_201712.npy'))
fm1711_id = list(np.load('../input/card_id_test_first_active_201711.npy'))
fm1710_id = list(np.load('../input/card_id_test_first_active_201710.npy'))
fm_id_list_10_12 = fm1712_id + fm1711_id + fm1710_id

# 201710 ~ 201712はエキスパートモデルの予測値に置き換える
submit.loc[submit[key].isin(fm_id_list_10_12), target] = base.loc[base[key].isin(fm_id_list_10_12), 'prediction'].values

#  for i in range(1, 10, 1):
#      tmp_fm_id = list(np.load(f'../input/card_id_test_first_active_20170{i}.npy'))
#      base_pred = submit.loc[submit[key].isin(tmp_fm_id), target] * (10-i)*0.1
#      ex_pred = base.loc[base[key].isin(tmp_fm_id), 'prediction'] * (i)*0.1
#      submit.loc[submit[key].isin(tmp_fm_id), target] = base_pred.values + ex_pred.values

submit.to_csv('../submit/0126_110_submit_blender_LB3-686_stack_fm_201712_expert_pred_10~12_only.csv', index=False)
