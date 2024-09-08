#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=test_cuda
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=35
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100G
#SBATCH --time=2-00:00

source ~/.bashrc

module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

/scratch/prj/inf_du/ziyan/benchmarking_meltingpot/bk_conda/bin/python /scratch/prj/inf_du/ziyan/benchmarking_meltingpot/Benchmarking-Meltingpot/examples/rllib/self_play_train_train.py --use_wandb 1