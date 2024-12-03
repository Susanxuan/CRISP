#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2022.10 
source activate /ailab/group/groups/aim/liuxinyuan/.conda/envs/scgpt
for sd in 1327; do
    for sp in split split2 split3; do
        for MMD in 0.0001; do
            python ../../CRISP/train_script.py \
            --config ../configs/sci.yaml \
            --split $sp \
            --seed $sd \
            --savedir ../results/sci_newdrugemb/sci_${sp}_${sd} \
            --MMD $MMD 
        done;
    done;
done