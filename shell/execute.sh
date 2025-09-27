#!/bin/bash

#PBS -V
#PBS -S /bin/bash
#PBS -j oe
#PBS -o path_for_logs
#PBS -N randomized
#PBS -q cpu-huge
#PBS -l select=1:ncpus=96:mem=512gb:ompthreads=1,walltime=96:00:00

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd $PBS_O_WORKDIR
uv run $MYFILE $MYARGS
