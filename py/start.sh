#!/bin/bash

python s012_auto_decrease.py lgb 0.3 10 7 1
python s013_auto_feat_increase.py lgb 0.3 10 5
python s012_auto_decrease.py lgb 0.3 10 7 1
python s013_auto_feat_increase.py lgb 0.3 10 5
python s012_auto_decrease.py lgb 0.1 50 5
python s013_auto_feat_increase.py lgb 0.1 50 3 
python s012_auto_decrease.py lgb 0.1 150 3
# python s013_auto_feat_increase.py lgb 0.02 150 3 
# python s012_auto_decrease.py lgb 0.02 150 3 
# python s001_lgb_main.py lgb 0.01 200 10 16 0

wait
sudo shutdown -P
