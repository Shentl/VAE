#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs_1/%j.out
#SBATCH --error=slurm_logs_1/%j.err

module load miniconda3
source activate
conda activate common

echo "Baseline"
echo "--batch_size 64 --z_dim 1 --l1 2 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 1 --l1 1 --epoch 100 --save z_dim_2
echo "--batch_size 64 --z_dim 2 --l1 1 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 2 --l1 1 --epoch 100 --save z_dim_1

echo "DVAE"
echo "Add_noise --batch_size 64 --z_dim 1 --l1 2 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 1 --l1 1 --epoch 100 --save z_dim_2 --add_noise
echo "Add_noise --batch_size 64 --z_dim 2 --l1 1 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 2 --l1 1 --epoch 100 --save z_dim_1 --add_noise

echo "BN VAE"
echo "Add_BN --batch_size 64 --z_dim 1 --l1 2 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 1 --l1 1 --epoch 100 --save z_dim_2 --add_BN
echo "Add_BN --batch_size 64 --z_dim 2 --l1 1 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 2 --l1 1 --epoch 100 --save z_dim_1 --add_BN

