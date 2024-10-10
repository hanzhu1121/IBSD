#!/bin/bash
python3 $2 --result_dir $1 --pretrain $3 --gpus $4 2>&1 | tee ./logs/TestLogs/$1.txt

