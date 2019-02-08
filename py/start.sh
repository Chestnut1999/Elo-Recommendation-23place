#!/bin/bash

# python s005_lgb_outlier_classifier.py 0.01 10 
# python s001_lgb_main.py 0.01 0 10 term
python s019_arange_dist_train.py 13 5 5 term
python s019_arange_dist_train.py 13 6 5 term
python s019_arange_dist_train.py 13 9 5 term
python s019_arange_dist_train.py 13 15 5 term
python s019_arange_dist_train.py 13 18 5 term
python s019_arange_dist_train.py 13 24 5 term
python s003_lgb_params_tune.py


# python s019_arange_dist_train.py 45 2016-08 5 pmo

# wait
# sudo shutdown -P
