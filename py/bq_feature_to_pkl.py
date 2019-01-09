import numpy as np
import pandas as pd
import os
HOME = os.path.expanduser("~")
import sys
sys.path.append(f"{HOME}/kaggle/data_analysis/library")
import utils
import glob
import re


path_list = glob.glob(f'../features/bigquery/*905*auth_1*')

for path in path_list:

    #  fname = 'his_' + re.search(r'his_([^/.]*).csv', path).group(1)
    fname = re.search(r'feat([^/.]*)_auth', path).group(1)
    feat_no = f"{fname}_au1_"
    df = pd.read_csv(path)

    base = utils.read_df_pkl('../input/base0*')
    base = base.merge(df, how='left', on='card_id')
    train = base[~base['target'].isnull()]
    test = base[base['target'].isnull()]

    for col in df.columns:
        if col.count('__'):
            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col.replace('__', '@').replace('his_', '')}@",
                              obj=train[col].values)
            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col.replace('__', '@').replace('his_', '')}@",
                              obj=test[col].values)
        else:
            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col}@",
                              obj=train[col].values)
            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col}@",
                              obj=test[col].values)
