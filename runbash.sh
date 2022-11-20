#!/bin/bash
echo Your container args are: "$@"
echo "$1"
echo "$2"

python3 que2.py --clf_name $1 --random_state $2