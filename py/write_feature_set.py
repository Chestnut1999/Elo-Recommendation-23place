import sys
import numpy as np
import pandas as pd
import glob
import re
import os


feature_num = 300


if not(os.path.exists('../features/save_feature_set.csv')):

    base = pd.Series(np.zeros(feature_num))

    path_list = glob.glob('../features/4_winner/*.gz')
    feature_list = []

    for path in path_list:
        fname = re.search(r'/([^/.]*).gz', path).group(1)
        feature_list.append(fname)
    sr_feature = pd.Series(feature_list, name=sys.argv[1])

    ftable = pd.concat([base, sr_feature], axis=1)

    ftable.to_csv('../features/save_feature_set.csv', index=False)

else:
    ftable = pd.read_csv('../features/save_feature_set.csv')

    path_list = glob.glob('../features/4_winner/*.gz')
    feature_list = []

    for path in path_list:
        fname = re.search(r'/([^/.]*).gz', path).group(1)
        feature_list.append(fname)
    sr_feature = pd.Series(feature_list, name=sys.argv[1])

    ftable = pd.concat([ftable, sr_feature], axis=1)

    ftable.to_csv('../features/save_feature_set.csv', index=False)


