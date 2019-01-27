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
#  fm201712_all = utils.read_pkl_gzip('../model/201712/stack/0126_0933_elo_first_month201712_all_dist_all_03_stack_1seed_lr0-02_round75000_CV3-6547.gz')
#  fm201712_org = utils.read_pkl_gzip('../model/201712/stack/0126_0933_elo_first_month201712_org0_dist_all_03_stack_1seed_lr0-02_round75000_CV3-7252.gz')
fm201712 = utils.read_pkl_gzip('../stack/0127_120_stack_lgb_lr0.01_349feats_1seed_31leaves_iter3915_OUT0_CV1-139620018388889_LB.gz').set_index(key)
fm201711 = utils.read_pkl_gzip('../stack/').set_index(key)
fm201710 = utils.read_pkl_gzip('../stack/').set_index(key)
#========================================================================

base.set_index(key, inplace=True)
base['pred_17-12'] = fm201712['prediction']
base['pred_17-11'] = fm201711['prediction']
base['pred_17-10'] = fm201710['prediction']
base.reset_index(inplace=True)

submit = pd.read_csv('../log_submit/0117_081_elo_5blender_LB3.686.csv')

# First Month id list

# 201710 ~ 201712はエキスパートモデルの予測値に置き換える
fname_ym = ''
for ym in ['2017-12', '2017-11', '2017-10']:
    fm_id_list = base[base['first_active_month']==ym][key].values
    print( base.loc[base[key].isin(fm_id_list), f'pred_{ym[-5:]}'].head())
    submit.loc[submit[key].isin(fm_id_list), target] = base.loc[base[key].isin(fm_id_list), f'pred_{ym[-5:]}']
    print(submit.loc[submit[key].isin(fm_id_list), f'pred_{ym[-5:]}'].head())


submit.to_csv(f'../submit/{start_time[4:11]}_submit_blender_LB3-686_stack_FAM2017-10-12_expert_pred.csv', index=False)
