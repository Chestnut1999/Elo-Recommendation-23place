#!/bin/bash

# python s002_lgb_feat_selection.py lgb 0.01 200
# python s004_lgb_decrease.py
python s001_lgb_main.py lgb 0.01 200 10 16 1
python s001_lgb_main.py lgb 0.01 200 10 16 0

wait
sudo shutdown -P
