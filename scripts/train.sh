#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2022.10 
source activate perturb
for sd in 1327 1337 1347; do
    for sp in split split2 split3; do
        for MMD in 0.1; do
            python /ailab/user/liuxinyuan/projects/perturb_project/finetuning/CRISP/train_script.py \
            --config /ailab/user/liuxinyuan/projects/perturb_project/finetuning/benchmark/CRISP/configs/nips_mycpa.yaml \
            --split $sp \
            --seed $sd \
            --savedir /ailab/user/liuxinyuan/projects/perturb_project/finetuning/benchmark/CRISP/results/nips_adapt/nips_${sp}_${sd} \
            --MMD $MMD
        done;
    done;
done