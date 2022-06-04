#!/bin/bash

# Snuba sweep over all datasets and embeddings
resdir=results/neurips2022/iws
mkdir -p ${resdir}

snuba_cardinality=1
n_labeled_points=100
lf_selector=iws

for seed in 0
do 
    for emb in raw pca resnet18 vae openai
    do 
        for dataset in mnist cifar10 spherical_mnist permuted_mnist ecg 
        do
        # TODO add navier stokes

            savedir=${resdir}/${seed}/${emb}/${dataset}
            mkdir -p ${savedir}

            python fwrench/applications/pipeline.py \
                --lf_selector=${lf_selector} \
                --n_labeled_points=${n_labeled_points} \
                --snuba_cardinality=${snuba_cardinality} \
                --seed=${seed} \
                --embedding=${emb} \
                --dataset=${dataset} \
                |& tee -a ${savedir}/res_seed${seed}.log

        done
    done
done
