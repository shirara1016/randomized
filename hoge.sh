#!/bin/bash

for method in randomized; do
    for delta in 0.0 0.1 0.2 0.3 0.4 0.5; do
        MYFILE=exp.py
        MYARGS="--method $method --delta $delta"
        qsub -v "MYFILE=$MYFILE,MYARGS=$MYARGS" shell/execute.sh
  done
done
