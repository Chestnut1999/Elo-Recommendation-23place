# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import sys
import re
import gc
import glob

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

path_list = glob.glob('../stack/*.gz')
import pickle
import datetime
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

blend_list = []
score_list = []
for path in path_list:
    try:
        tmp = utils.read_pkl_gzip(path)
        if tmp.shape[0]!=325540:
            continue
        try:
            score = float(re.search(r"CV([^.]*)_", path).group(1).replace('-', '.'))
        except AttributeError:
            continue

        if score>3.69:
            continue
        score_list.append(score)
        blend_list.append(path)
    except pickle.UnpicklingError:
        pass

logger.info(f"Blend List: {len(blend_list)}")

sum_score = np.sum(score_list)
result = pd.DataFrame()
for path, score in zip(blend_list, score_list):
    tmp = utils.read_pkl_gzip(path)

    if len(result):
        result['prediction'] += tmp['prediction'] * (score/sum_score)
    else:
        result = tmp.copy()
        result['prediction'] *= (score/sum_score)


submit_id = pd.read_csv('../input/sample_submission.csv')['card_id'].values
test = result.set_index('card_id').loc[submit_id, :].reset_index()[['card_id', 'prediction']].rename(columns={'prediction':'target'})
test.to_csv(f'../submit/{start_time[4:12]}_elo_{len(blend_list)}blender.csv', index=False)
