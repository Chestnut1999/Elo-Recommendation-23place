import sys
import pandas as pd

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'column_0']

win_path = f'../features/4_winner/*.gz'
#  win_path = f'../model/old_201712/*.gz'
import numpy as np
import datetime
import glob
import gc
import os
from sklearn.metrics import mean_squared_error
HOME = os.path.expanduser('~')

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


#========================================================================
# Data Load
base_path = glob.glob('../features/0_base/*.gz')
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path) + glob.glob('base_path')
tmp_path_list = glob.glob('../features/5_tmp/*.gz')
win_path_list += tmp_path_list

base = utils.read_df_pkl('../input/base*')

base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
feature_list = utils.parallel_load_data(path_list=win_path_list)
df_feat = pd.concat(feature_list, axis=1)
train = pd.concat([base_train, df_feat.iloc[:len(base_train), :]], axis=1)
test = pd.concat([base_test, df_feat.iloc[len(base_train):, :].reset_index(drop=True)], axis=1)
train_test = pd.concat([train, test], axis=0)

#========================================================================
# card_id list by first active month
try:
    sys.argv[5]
    train_latest_id_list = np.load('../input/card_id_train_first_active_201712.npy')
    test_latest_id_list = np.load('../input/card_id_test_first_active_201712.npy')
    train = train.loc[train[key].isin(train_latest_id_list), :].reset_index(drop=True)
    test = test.loc[test[key].isin(test_latest_id_list), :].reset_index(drop=True)
    submit = []
except IndexError:
    pass
#========================================================================

model_list = utils.read_pkl_gzip('../model/201712/0122_elo_first_month201712_10seed_fold_model_list.gz')
use_cols = pd.read_csv('../model/201712/0122_elo_first_month201712_fold_model_use_cols.csv').values.reshape(-1,)

pred = np.zeros(len(train_test))
for model in model_list:
    pred += model.predict(train_test[use_cols])
pred /= len(model_list)

feature_name = '014_l02_elo_first_month201712_prediction'
utils.to_pkl_gzip(obj=pred, path='../features/5_tmp/{feature_name}')
