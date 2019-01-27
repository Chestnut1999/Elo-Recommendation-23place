#!/bin/bash

python s019_arange_dist_train.py '../model/201712/200lag/*.gz' 2017-12 10 pm
python s019_arange_dist_train.py '../model/201711/200lag/*.gz' 2017-11 10 pm
python s019_arange_dist_train.py '../model/201710/200lag/*.gz' 2017-10 10 pm

# wait
# sudo shutdown -P
