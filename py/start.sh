#!/bin/bash

for i in `seq 1 2000`
do
    echo $i
    python s104_lgb_ensemble.py lgb 0.05 $i
done

# wait
# sudo shutdown -P
