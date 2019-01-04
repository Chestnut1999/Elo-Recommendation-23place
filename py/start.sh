#!/bin/bash

python s002_lgb_feat_selection.py lgb 0.01 200
python s004_lgb_decrease.py

wait
sudo shutdown -P
