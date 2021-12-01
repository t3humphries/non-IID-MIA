#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

RUNS=10
########### Generate all datasets #########
DATASETS=('adult-clus2-0' 'compas-clus2-2' 'adult-numedu' 'adult-sex' 'heart-mix' 'students2' 'kdd-census2' 'texas-region-3')
for DATASET in "${DATASETS[@]}"
do
	echo $DATASET
	python3.8 -u preprocess_data.py $DATASET
done

######### Generate splits #################
# Cluster splits
python3.8 create_splits.py 'adult-clus2-0' 'g0_split' --nsamples=10000 --nruns=$RUNS
python3.8 create_splits.py 'adult-clus2-0' 'g1_split' --nsamples=10000 --nruns=$RUNS
python3.8 create_splits.py 'compas-clus2-2' 'g0_split' --nsamples=2000 --nruns=$RUNS
python3.8 create_splits.py 'compas-clus2-2' 'g1_split' --nsamples=2000 --nruns=$RUNS
# Attribute splits
for PVAL in 0.00 0.10 0.25 0.50 0.75 0.90 1.00
do
 python3.8 create_splits.py 'adult-numedu' 'p_split' --nsamples=10000 --pval=$PVAL --nruns=$RUNS
 python3.8 create_splits.py 'adult-sex' 'p_split' --nsamples=10000 --pval=$PVAL --nruns=$RUNS
done
# Dataset splits
python3.8 create_splits.py 'heart-mix' 'g_full_split' --nruns=$RUNS
python3.8 create_splits.py 'students2' 'g_full_split' --nruns=$RUNS
python3.8 create_splits.py 'kdd-census2' 'g0_split' --nsamples=10000 --nruns=$RUNS
#python3.8 create_splits.py 'kdd-census2' 'g1_split' --nsamples=10000 --nruns=$RUNS
python3.8 create_splits.py 'texas-region-3' 'g0_split' --nsamples=10000 --nruns=$RUNS
#python3.8 create_splits.py 'texas-region-3' 'g1_split' --nsamples=10000 --nruns=$RUNS

############### Run experiments ###################
python3.8 -u attack.py 'heart-mix' --target_batch_size=32
python3.8 -u attack.py 'students2' --target_batch_size=32
DATASETS=('adult-clus2-0' 'compas-clus2-2' 'adult-numedu' 'adult-sex' 'kdd-census2' 'texas-region-3')
for DATASET in "${DATASETS[@]}"
do
  python3.8 -u attack.py $DATASET
done
