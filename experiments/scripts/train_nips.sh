#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2022.10 
source activate /ailab/group/groups/aim/liuxinyuan/.conda/envs/scgpt
for sd in 1327; do
    for sp in split split2 split3; do
        for MMD in 0.1; do
            python ../../CRISP/train_script.py \
            --config ../configs/nips.yaml \
            --split $sp \
            --seed $sd \
            --savedir ../results/nips_newdrugemb/nips_${sp}_${sd} \
            --MMD $MMD
        done;
    done;
done