import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

#========================================================================
# ods.ai 3rd kernel
# https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78903
# KFold: n_splits=6(or 7)!, shuffle=False!
def ods_kfold(train=[], seed=1208, fold=6, key='card_id', target='target'):

    #  ========================================================================
    #  ods.ai 3rd kernel
    #  https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/78903
    #  KFold: n_splits=6(or 7)!, shuffle=False!
    #  ========================================================================
    print(train.shape)
    train['rounded_target'] = train['target'].round(0)
    train = train.sort_values('rounded_target').reset_index(drop=True)
    vc = train['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df = pd.DataFrame()
    train['indexcol'],idx = 0,1
    for k,v in vc.items():
        step = train.shape[0]/v
        indent = train.shape[0]/(v+1)
        df2 = train[train['rounded_target'] == k].sample(v, random_state=seed).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j*step + 0.000001*idx
        df = pd.concat([df2,df])
        idx+=1
    train = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del train['indexcol'], train['rounded_target']
    folds = KFold(n_splits=fold, shuffle=False, random_state=seed)
    kfold = list(folds.split(train, train[target].values))

    # card_id listにする
    trn_list = []
    val_list = []
    for trn, val in kfold:
        trn_ids = df.iloc[trn][key].values
        val_ids = df.iloc[val][key].values
        trn_list.append(trn_ids)
        val_list.append(val_ids)
    fold_list = [trn_list, val_list]

    return fold_list
#========================================================================
