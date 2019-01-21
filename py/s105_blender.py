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
from utils import logger_func
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

path_list = glob.glob('../stack/*.gz')
import pickle
import datetime
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

blend_path = glob.glob('../ensemble/*.csv')
blend_list = []

for path in blend_path:
    elem = pd.read_csv(path)
    blend_list.append(elem.copy())

blending = np.zeros(len(elem))
for elem in blend_list:
    pred = elem['target']
    blending += pred
blending /= len(blend_list)

submit = pd.read_csv('../input/sample_submission.csv')
submit['target'] = blending

clf = utils.read_pkl_gzip('../stack/0112_155_outlier_classify_9seed_lgb_binary_CV0-9047260065151934_200features.gz')
clf = clf.iloc[-len(submit):, ].reset_index(drop=True)
submit.loc[clf.prediction>0.45, 'target'] = -33.1


submit.to_csv(f'../submit/{start_time[4:12]}_elo_{len(blend_list)}blender_outlier_clf0.45_postprocessing.csv', index=False)
