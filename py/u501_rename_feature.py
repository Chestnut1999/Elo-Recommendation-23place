import glob
import os
import sys
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
import utils

path_list = glob.glob('../features/1_first_valid/*.gz')

key = ''
old_key = '_pts_'
new_key = '_pst_'

for path in path_list:

    if (path.count(key)):
    #  if not(path.count(key)):
    #  if not(path.count(key)):
        feature = utils.read_pkl_gzip(path)

        rename_path = path.replace(old_key, new_key).replace('.gz', '').replace('.gz', '').replace('.gz', '')
        utils.to_pkl_gzip(obj=feature, path=rename_path)
        os.system(f'rm {path}')

    else:
        feature = utils.read_pkl_gzip(path)
        rename_path = path.replace('.gz', '').replace('.gz', '').replace('.gz', '')
        utils.to_pkl_gzip(obj=feature, path=rename_path)
        os.system(f'rm {path}')
