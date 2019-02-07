#!/bin/bash

python s001_lgb_main.py 0.01 0 10 ods
python s003_lgb_params_tune.py

# python s019_arange_dist_train.py 45 2016-08 5 pmo

# wait
# sudo shutdown -P
