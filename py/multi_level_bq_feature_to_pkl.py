import numpy as np
import pandas as pd
import os
HOME = os.path.expanduser("~")
import sys
sys.path.append(f"{HOME}/kaggle/data_analysis/library")
import utils
import glob
import re


path_list = glob.glob(f'../features/bigquery/*auth_0*year*')

for path in path_list:

    #  fname = 'his_' + re.search(r'his_([^/.]*).csv', path).group(1)
    fname = re.search(r'feat([^/.]*)_auth', path).group(1)
    feat_no = f"{fname}_au0_"
    df = pd.read_csv(path)

    if path.count('year'):
        print( np.unique(df['year'].values))
        sys.exit()
        for year in np.unique(df['year'].values):

            base = utils.read_df_pkl('../input/base0*')
            base = base.merge(df.query(f"year=={year}"), how='left', on='card_id')
            train = base[~base['target'].isnull()]
            test = base[base['target'].isnull()]

            for col in df.columns:
                if col.count('__'):
                    utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col.replace('__', '@').replace('his_', '')}_year{year}", obj=train[col].values)
                    utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col.replace('__', '@').replace('his_', '')}_year{year}", obj=test[col].values)

    else:
        if path.count('dow') and path.count('timezone'):

            for month in np.unique(df['latest_month_no'].values):
                for dow in np.unique(df['dow'].values):
                    for timezone in np.unique(df['timezone'].values):

                        base = utils.read_df_pkl('../input/base0*')
                        base = base.merge(df.query(f"latest_month_no=={month}").query(f"dow=={dow}").query(f"timezone=='{timezone}'"), how='left', on='card_id')
                        train = base[~base['target'].isnull()]
                        test = base[base['target'].isnull()]

                        for col in df.columns:
                            if col.count('__'):
                                utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col.replace('__', '@').replace('his_', '')}_month{month}_dow{dow}_timezone{timezone}",
                                                  obj=train[col].values)
                                utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col.replace('__', '@').replace('his_', '')}_month{month}_dow{dow}_timezone{timezone}",
                                                  obj=test[col].values)

        elif path.count('dow') and not(path.count('timezone')):

            for month in np.unique(df['latest_month_no'].values):
                for dow in np.unique(df['dow'].values):

                    base = utils.read_df_pkl('../input/base0*')
                    base = base.merge(df.query(f"latest_month_no=={month}").query(f"dow=={dow}"), how='left', on='card_id')
                    train = base[~base['target'].isnull()]
                    test = base[base['target'].isnull()]

                    for col in df.columns:
                        if col.count('__'):
                            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col.replace('__', '@').replace('his_', '')}_month{month}_dow{dow}",
                                              obj=train[col].values)
                            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col.replace('__', '@').replace('his_', '')}_month{month}_dow{dow}",
                                                  obj=test[col].values)

        elif not(path.count('dow')) and path.count('timezone'):

            for month in np.unique(df['latest_month_no'].values):
                for timezone in np.unique(df['timezone'].values):

                    base = utils.read_df_pkl('../input/base0*')
                    base = base.merge(df.query(f"latest_month_no=={month}").query(f"timezone=='{timezone}'"), how='left', on='card_id')
                    train = base[~base['target'].isnull()]
                    test = base[base['target'].isnull()]

                    for col in df.columns:
                        if col.count('__'):
                            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col.replace('__', '@').replace('his_', '')}_month{month}_timezone{timezone}", obj=train[col].values)
                            utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col.replace('__', '@').replace('his_', '')}_month{month}_timezone{timezone}", obj=test[col].values)

        elif not(path.count('dow')) and not(path.count('timezone')):

            for month in np.unique(df['latest_month_no'].values):

                base = utils.read_df_pkl('../input/base0*')
                base = base.merge(df.query(f"latest_month_no=={month}"), how='left', on='card_id')
                train = base[~base['target'].isnull()]
                test = base[base['target'].isnull()]

                for col in df.columns:
                    if col.count('__'):
                        utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}train_{col.replace('__', '@').replace('his_', '')}_month{month}", obj=train[col].values)
                        utils.to_pkl_gzip(path=f"../features/1_first_valid/{feat_no}test_{col.replace('__', '@').replace('his_', '')}_month{month}", obj=test[col].values)

