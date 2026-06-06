#!/bin/bash
export LD_LIBRARY_PATH=/data/anaconda3/lib:$LD_LIBRARY_PATH
cd /data/wangye/trainable-openclaw
/data/anaconda3/bin/python3 -u tmp/baseline_eval.py > /tmp/baseline_output.log 2>&1
echo "EXIT_CODE=$?" >> /tmp/baseline_output.log
