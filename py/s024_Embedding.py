import numpy as np
import datetime
import glob
import gc
import os
import sys
import pandas as pd
from sklearn.decomposition import PCA

#========================================================================
# Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month', 'index', 'personal_term']

HOME = os.path.expanduser('~')
sys.path.append(f'{HOME}/kaggle/data_analysis/model')

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping
from utils import logger_func
logger=logger_func()

from dimensionality_reduction import go_bhtsne, UMAP


#========================================================================
# Data Load

win_path = f'../features/4_winner/*.gz'
#  win_path_list = glob.glob(win_path)
win_path_list = glob.glob(f'../features/0_exp/*.gz')

base = utils.read_df_pkl('../input/base_term*')
feature_list = utils.parallel_load_data(path_list=win_path_list)
df_feat = pd.concat(feature_list, axis=1)

#========================================================================

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
prefix = f"224_emb_{start_time[4:12]}"
feat_list = [col for col in df_feat.columns]
emb_list = []
feat_num = 10
ignore_list = [key, target, 'merchant_id', 'first_avtive_month', 'index']


for i in range(10):
    np.random.seed(i)
    tmp_list = np.random.choice(feat_list, feat_num)

    #========================================================================
    # Impute
    for col in tmp_list:
        if col in ignore_list:
            continue

        df_feat[col] = df_feat[col].replace(-1*np.inf, np.nan)
        df_feat[col] = df_feat[col].replace(1*np.inf, np.nan)
        if col.count('date'):
            val_min = df_feat[col].min()
            df_feat[col].fillna(val_min-100, inplace=True)
        elif col.count('month_diff'):
            val_min = df_feat[col].min()
            df_feat[col].fillna(val_min-10, inplace=True)
        else:
            df_feat[col].fillna(-1, inplace=True)

    #========================================================================


#     embeddeing = go_bhtsne(logger=logger, data=df_feat[tmp_list], D=2)
    embedding = UMAP(logger=logger, data=df_feat[tmp_list], D=2)
    emb_list.append(embedding)

    pca = PCA(n_components=0.81,whiten=True)
    feature = pca.fit_transform(pd.DataFrame(embedding, columns=['D1', 'D2']))

    if feature.ravel().shape[0]>500000:
        feat_1 = feature[:, 0].astype('float32')
        feat_2 = feature[:, 1].astype('float32')
        utils.to_pkl_gzip(obj=feat_1, path=f'../features/1_first_valid/{prefix}_PCA_D1_UMAP_seed{i}')
        utils.to_pkl_gzip(obj=feat_2, path=f'../features/1_first_valid/{prefix}_PCA_D2_UMAP_seed{i}')
    else:
        feature = feature.ravel().astype('float32')
        utils.to_pkl_gzip(obj=feature, path=f'../features/1_first_valid/{prefix}_PCA_UMAP_seed{i}')

