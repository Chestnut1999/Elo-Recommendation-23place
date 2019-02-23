import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import time, psutil, json, os, gc, datetime
from sklearn.metrics import mean_squared_error, roc_auc_score

import os
import sys
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
import utils


def rmse(y_true, y_pred):
	score = np.sqrt(mean_squared_error(y_true, y_pred))
	return score


#3.631661
#3.631645
#3.606998

print("Started")
print(psutil.virtual_memory())

dfx = pd.read_csv('../input/train.csv', usecols=['card_id','target'])

#dd = pd.read_csv('../log_submit/alijs_submit/cv_stack1_preds_01232204.csv')[['card_id','preds']]
#dd = pd.read_csv('../log_submit/alijs_submit/cv_stack1_preds_02112249.csv')[['card_id','preds']]
dd = pd.read_csv('../log_submit/alijs_submit/cv_stack1_preds_02192033.csv')[['card_id','preds']]
print(dd.head())
dfx = dfx.merge(dd)
dfx['error'] = (dfx['target'] - dfx['preds']) ** 2

#dd = pd.read_csv('../log_submit/alijs_submit/cv_o1_preds_12282024.csv')[['card_id','preds']]
#dd = pd.read_csv('../log_submit/alijs_submit/cv_o1_preds_02022141.csv')[['card_id','preds']]

#dd = pd.read_csv('../log_submit/alijs_submit/cv_o1_preds_02101601.csv')[['card_id','preds']] #1.544343
dd = pd.read_csv('../log_submit/alijs_submit/cv_stack1_o_preds_02192129.csv')[['card_id','preds']] # 1.5427064
dd.columns = ['card_id','predswo1']
print(dd.head())
dfx = dfx.merge(dd)

#dd2 = pd.read_csv('../log_submit/alijs_submit/cv_o1_preds_02131518.csv')[['card_id','preds']] #1.544674
dd2 = pd.read_csv('../log_submit/alijs_submit/cv_stack1_o_preds_02192129.csv')[['card_id','preds']] # 1.5427064

dd2.columns = ['card_id','predswo2']
print(dd2.head())
dfx = dfx.merge(dd2)

dfx['predswo'] = dfx['predswo1']*0.5 + dfx['predswo2']*0.5

dfx['errorwo'] = (dfx['target'] - dfx['predswo']) ** 2
dfx['errorwo1'] = (dfx['target'] - dfx['predswo1']) ** 2
dfx['errorwo2'] = (dfx['target'] - dfx['predswo2']) ** 2

#  ft = pd.read_pickle('ft_refminmax5.pkl')
#  print(ft['type'].value_counts())
#  dfx = dfx.merge(ft, how='left')
#print(dfx)

ft = pd.read_csv('../input/card_ids_grouping.csv')[['card_id','type']]
#  ft2 = pd.read_csv('0219_go_elo_classifier_pred_NoOutlierFlg.csv')[['card_id','no_out_flg','clf_pred']]
ft2 = utils.read_pkl_gzip('../input/base_no_out_clf.gz')[['card_id','no_out_flg','clf_pred']]
dfx = dfx.merge(ft, how='left').merge(ft2, how='left')
#print(dfx)

dfx['targeto'] = dfx['target'].apply(lambda x: 1 if x < -20 else 0)
print(dfx.groupby(['no_out_flg','type'])['targeto'].agg(['mean','sum','size']).reset_index())

print('error preds',rmse(dfx['target'],dfx['preds']))
print('error wo1   ',rmse(dfx['target'],dfx['predswo1']))
print('error wo2   ',rmse(dfx['target'],dfx['predswo2']))
print('error wo   ',rmse(dfx['target'],dfx['predswo']))

sel = (dfx['type'] == 0)
#dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predsmo']*0.2 + dfx.loc[sel, 'preds']*0.8)
sel = (dfx['type'] == 2)
dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predswo']*0.65 + dfx.loc[sel, 'preds']*0.35)
#dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predswo']*0.35 + dfx.loc[sel, 'predsfo']*0.20 + dfx.loc[sel, 'preds']*0.45)
sel = (dfx['type'] == 1)
dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predswo']*0.55 + dfx.loc[sel, 'preds']*0.45)
sel = (dfx['no_out_flg'] == 1)
dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predswo']*0.7 + dfx.loc[sel, 'preds']*0.3)
sel = (dfx['clf_pred'] < 0.011)
dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predswo']*0.7 + dfx.loc[sel, 'preds']*0.3)
sel = (dfx['clf_pred'] < 0.014) & (dfx['type'] > 0)
dfx.loc[sel, 'preds'] = (dfx.loc[sel, 'predswo']*0.9 + dfx.loc[sel, 'preds']*0.1)
sel = (dfx['type'] == 10)
dfx.loc[sel, 'preds'] = dfx.loc[sel, 'predswo']

print('error updat',rmse(dfx['target'],dfx['preds']))
#dfx.loc[dfx['type'] == 0, 'preds'] /= 1.01
#print('error updat2',rmse(dfx['target'],dfx['preds']))

# dfx[['card_id','preds']].to_csv('../output/alijs_current_best_cv.csv', index=False)
dfx.rename(columns={'preds':'prediction'}, inplace=True)
result_list = []
result_list.append(dfx[['card_id', 'prediction']])
print(dfx.shape)

if 1:
    print('\n======== prepare submit')
    dfx = pd.read_csv('../input/test.csv', usecols=['card_id'])

    dd = pd.read_csv('../log_submit/alijs_submit/stack1_preds_02192033.csv')[['card_id','target']]
    print(dd.head())
    dfx = dfx.merge(dd)

    #dd = pd.read_csv('../log_submit/alijs_submit//o1_preds_02022141.csv', compression='csv')[['card_id','target']]
    #dd = pd.read_csv('../log_submit/alijs_submit//o1_preds_02101601.csv', compression='csv')[['card_id','target']]
    dd = pd.read_csv('../log_submit/alijs_submit/stack1_o_preds_02192129.csv')[['card_id','target']]
    dd.columns = ['card_id','predswo1']
    print(dd.head())
    dfx = dfx.merge(dd)

    #dd2 = pd.read_csv('../log_submit/alijs_submit//o1_preds_02131518.csv', compression='csv')[['card_id','target']] #1.544674
    dd2 = pd.read_csv('../log_submit/alijs_submit/stack1_o_preds_02192129.csv')[['card_id','target']] #1.544674
    dd2.columns = ['card_id','predswo2']
    print(dd2.head())
    dfx = dfx.merge(dd2)

    dfx = dfx.merge(ft, how='left')
    #print(dfx)
    dfx = dfx.merge(ft2, how='left')

    dfx['predswo'] = dfx['predswo1']*0.5 + dfx['predswo2']*0.5

    sel = (dfx['type'] == 0)
    #dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predsmo']*0.2 + dfx.loc[sel, 'target']*0.8)
    sel = (dfx['type'] == 2)
    dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predswo']*0.65 + dfx.loc[sel, 'target']*0.35)
    #dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predswo']*0.35 + dfx.loc[sel, 'predsfo']*0.20 + dfx.loc[sel, 'target']*0.45)
    sel = (dfx['type'] == 1)
    dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predswo']*0.55 + dfx.loc[sel, 'target']*0.45)
    sel = (dfx['no_out_flg'] == 1)
    dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predswo']*0.7 + dfx.loc[sel, 'target']*0.3)
    sel = (dfx['clf_pred'] < 0.011)
    dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predswo']*0.7 + dfx.loc[sel, 'target']*0.3)
    sel = (dfx['clf_pred'] < 0.014) & (dfx['type'] > 0)
    dfx.loc[sel, 'target'] = (dfx.loc[sel, 'predswo']*0.9 + dfx.loc[sel, 'target']*0.1)
    sel = (dfx['type'] == 10)
    dfx.loc[sel, 'target'] = dfx.loc[sel, 'predswo']
    print(dfx.head())

    now = str(datetime.datetime.now().strftime("%m%d%H%M"))
    tf = f'../output/{now[0:11]}_alijs_stack_submit_pp_%s.csv' % now
#     dfx[['card_id','target']].to_csv(tf, index=False)
    dfx.rename(columns={'target':'prediction'}, inplace=True)
    result_list.append(dfx[['card_id', 'prediction']])
    print('saved to file:',tf)
dfx = pd.concat(result_list , axis=0)
dfx.to_csv(tf, index=False)
print("Done.")
