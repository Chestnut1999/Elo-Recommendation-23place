#!/bin/bash

python s001_lgb_main.py 0.01 0 10 ods all 4
python s003_lgb_params_tune.py

# wait
# sudo shutdown -P
