import gc
import re
import pandas as pd
import numpy as np
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)
import os
import sys
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
import utils
from utils import get_categorical_features, get_numeric_features, reduce_mem_usage, elo_save_feature
from preprocessing import get_dummies
import datetime

from tqdm import tqdm
import time
import sys
from joblib import Parallel, delayed

key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_avtive_month']

df_hist = pd.read_csv('../input/historical_transactions.csv')
df_hist['purchase_amount_new'] = np.round(df_hist['purchase_amount'] / 0.00150265118 + 497.06, 2)
df_hist['installments'] = df_hist['installments'].map(lambda x:  
                                                    1 if x<1 else
                                                    1 if x>100 else
                                                      x
                                                     )

#========================================================================
# Dataset Load 
use_cols = [key, 'authorized_flag', 'installments', 'merchant_category_id', 'merchant_id', 'subsector_id', 'month_lag', 'purchase_amount_new', 'purchase_date']
df_hist = df_hist[use_cols]
# df_new = df_new[use_cols]

auth1 = df_hist[df_hist['authorized_flag']=='Y']
# auth0 = df_hist[df_hist['authorized_flag']=='N']

# df_trans = pd.concat([auth1, df_new], axis=0)

def get_new_columns(name,aggs):
    if len(name):
        return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    else:
        return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
#========================================================================

df = auth1

#========================================================================
# Merchant id 別の集計
# month_lagで集計する（purchase_dateは別にやる）

def raddar_2level_agg(new_max, new_min, old_max, old_min):
    #========================================================================
    # Args Setting
    level = [key, 'merchant_id', 'month_lag']
    new_month_lag_max = new_max
    new_month_lag_min = new_min
    old_month_lag_max = old_max
    old_month_lag_min = old_min
    #========================================================================

    #========================================================================
    # Aggregation
    print("Aggregation Start!!")

    aggs = {}
    aggs['purchase_amount_new'] = ['sum']
    aggs['installments'] = ['mean', 'max', 'sum']
    
    df = auth1
    df_agg = df.groupby(level).agg(aggs)
    
    new_cols = get_new_columns(name='', aggs=aggs)
    df_agg.columns = new_cols
    df_agg[f'purchase_amount_new_sum_per_installments_sum'] = df_agg[f'purchase_amount_new_sum'] / df_agg[f'installments_sum']
    
    mer_cnt = df.groupby([key, 'merchant_id'])['month_lag'].nunique().reset_index().rename(columns={'month_lag':'month_lag_cnt'})
    df_agg = df_agg.reset_index().merge(mer_cnt, how='inner', on=[key, 'merchant_id'])
    #========================================================================
    
    #========================================================================
    # month_lag別に切り出して集計を行う
    print("Aggregate Term Setting")
    df_merchant = df_agg[df_agg['month_lag_cnt']>1]
    del df_agg
    gc.collect()
    df_merchant.drop(['month_lag_cnt'], axis=1, inplace=True)
    
    # 期間を絞る
    new_term = new_month_lag_max - new_month_lag_min
    old_term = old_month_lag_max - old_month_lag_min
    new = df_merchant[df_merchant['month_lag']<= new_month_lag_max][df_merchant['month_lag']>= new_month_lag_min]
    old = df_merchant[df_merchant['month_lag']<= old_month_lag_max][df_merchant['month_lag']>= old_month_lag_min]
    
    feat_cols = [col for col in df_merchant.columns if col.count('amount') or col.count('install')]
    aggs = {}
    for col in feat_cols:
        if col.count('install') and not(col.count('per')):
            aggs[col] = ['mean']
        else:
            aggs[col] = ['sum']
            
    # 複数month_lagをもつデータの場合は、集計する
    if new_term>0:
        new.groupby([key, 'merchant_id'])[feat_cols]
    else:
        new.set_index([key, 'merchant_id'], inplace=True)
    
    if old_term>0:
        old.groupby([key, 'merchant_id'])[feat_cols]
    else:
        old.set_index([key, 'merchant_id'], inplace=True)
    #========================================================================
        
    #========================================================================
    # oldに存在するがnewにいないcard_id, merchantをnewにもたせる。
    print("Get Lost Merchant and Card ID")
    new.reset_index(inplace=True)
    old.reset_index(inplace=True)
    new['flg'] = 1
    tmp_cols = [key, 'merchant_id']
    old_lost = old[tmp_cols].merge(new[tmp_cols + ['flg']], how='left', on=[key, 'merchant_id'])
    old_lost = old_lost[old_lost['flg'].isnull()]
    old_lost = old_lost[tmp_cols]
    new = pd.concat([new, old_lost], ignore_index=True)
    new.drop('flg', axis=1, inplace=True)
    #========================================================================
        
    #========================================================================
    # Make Ratio Feature
    print("Make Ratio Feature")
    fname = f'flag{new_month_lag_max}_{new_month_lag_min}-plag{old_month_lag_max}_{old_month_lag_min}'
    new = new.merge(old, how='left', on=[key, 'merchant_id'])
    for col in feat_cols:
       new[f"{fname}_{col}"]  = new[col+'_x'] / new[col+'_y']
       new[f"{fname}_{col}"].fillna(0, inplace=True)
    #========================================================================
    
    #========================================================================
    # card_id * merchant_id別のtop frequency ranking
    # all term version
    mer_cnt = df_hist.groupby([key, 'merchant_id'])['month_lag'].nunique().reset_index().rename(columns={'month_lag':'month_lag_cnt'})
    mer_cnt.sort_values(by=[key, 'month_lag_cnt'], ascending=False, inplace=True)
    mer_cnt = utils.row_number(df=mer_cnt, level=key)
    mer_cnt.set_index([key, 'merchant_id'], inplace=True)
    
    df_merchant = new.set_index(tmp_cols).join(mer_cnt).reset_index()
    del new, old, mer_cnt
    gc.collect()
    
    use_cols = [key, 'merchant_id'] + [col for col in df_merchant.columns if col.count('flag')] + ['month_lag_cnt', 'row_no']
    df_merchant = df_merchant[use_cols]
    #========================================================================
    
    
    #========================================================================
    # merchant_id別に集計を行ったら、
    # 1. それらを更に集計する. frequencyが高いmerchantのみで集計するパターンも作る
    # 2. frequencyの高いmerchnatでまとめてtop1~10カラムを作る ------->>> frequencyについては、全体と直近半年の両パターンでカウントし、特徴を作る
    #========================================================================
    prefix = '241_rad'
    print(f"{prefix} Feature Saving...")
    
    feat_cols = [col for col in df_merchant.columns if col.count('flag')]
    aggs = {}
    for col in feat_cols:
        aggs[col] = ['mean', 'max']
    
    df_agg = df_merchant[[key] + feat_cols].fillna(0).groupby(key)[feat_cols].agg(aggs)
    
    # Rename
    fname = 'auth1_all'
    new_cols = get_new_columns(name=fname, aggs=aggs)
    df_agg.columns = new_cols
    #========================================================================
    
    #========================================================================
    # Save Feature
    base = utils.read_df_pkl('../input/base_no_out*')
    base = base[[key, target]].set_index(key)
    base = base.join(df_agg)
    base.fillna(-1, inplace=True)
    del df_agg
    gc.collect()
    
    elo_save_feature(df_feat=base, prefix=prefix)
    print('Complete!')
    #========================================================================
    
    
    #========================================================================
    # frequencyで絞った場合
    prefix = '242_rad_freq_2Mover'
    print(f"{prefix} Feature Saving...")
    
    freq_high = df_merchant[df_merchant['month_lag_cnt']>2]
    df_agg = freq_high[[key] + feat_cols].fillna(0).groupby(key)[feat_cols].agg(aggs)
    
    # Rename
    fname = 'auth1_all'
    new_cols = get_new_columns(name=fname, aggs=aggs)
    df_agg.columns = new_cols
    #========================================================================
    
    #========================================================================
    # Save Feature
    base = utils.read_df_pkl('../input/base_no_out*')
    base = base[[key, target]].set_index(key)
    base = base.join(df_agg)
    base.fillna(-1, inplace=True)
    del df_agg
    gc.collect()
    
    elo_save_feature(df_feat=base, prefix=prefix)
    print('Complete!')
    #========================================================================


def main():

    pattern_list = [
        # One Month Ver
    #     [0, 0, -1, -1]
    #     ,[-1, -1, -2, -2]
    #     ,[-2, -2, -3, -3]

        # Two Month Ver
        #  [0, -1, -2, -3]
        #  ,[-1, -2, -3, -4]
        #  ,[-2, -3, -4, -5]

        #  # Three Month Ver
        #  ,[0, -2, -3, -5]
        #  ,[-1, -3, -4, -6]
        #  ,[-2, -4, -5, -7]

        #  # One / Two Month Ver
        #  ,[0, 0, -1, -2]
        #  ,[-1, -1, -2, -3]
        #  ,[-2, -2, -3, -4]

        #  # One / Three Month Ver
        #  ,[0, 0, -1, -3]
        #  ,[-1, -1, -2, -4]
        #  ,[-2, -2, -3, -5]

        #  # One / Six Month Ver
        #  ,[0, 0, -1, -6]
        #  ,[-1, -1, -2, -7]
        #  ,[-2, -2, -3, -8]

        #  # Two / Three Month Ver
        #  ,[0, -1, -2, -4]
        #  ,[-1, -2, -3, -5]
        [-2, -3, -4, -6]

        # Two / Six Month Ver
        ,[0, -1, -2, -7]
        ,[-1, -2, -3, -8]
        ,[-2, -3, -4, -9]

        # Three / Six Month Ver
        ,[0, -2, -3, -8]
        ,[-1, -3, -4, -9]
        ,[-2, -4, -5, -10]

    ]
    for pattern in pattern_list:
        raddar_2level_agg(*pattern)

if __name__=='__main__':
    main()
