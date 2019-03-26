#!/usr/bin/env bash

LR="0.001"
EPOCHS="20"
DATASETS=( "digits" "fashion" )
L1_VALS=( "0.001" "0.0001" )
L2_VALS=( "0.01" "0.001" )

for DATASET in "${DATASETS[@]}"
do
    python -m network-pruning.train --dataset=$DATASET --epochs=$EPOCHS --learning_rate=$LR --l1_reg=0 --l2_reg=0

    for L1 in "${L1_VALS[@]}"
    do
        python -m network-pruning.train --dataset=$DATASET --epochs=$EPOCHS --learning_rate=$LR --l1_reg=$L1 --l2_reg=0
    done

    for L2 in "${L2_VALS[@]}"
    do
        python -m network-pruning.train --dataset=$DATASET --epochs=$EPOCHS --learning_rate=$LR --l1_reg=0 --l2_reg=$L2
    done
done
