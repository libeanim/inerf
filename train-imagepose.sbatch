#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-2080ti
#SBATCH --time=2-00:00
#SBATCH --mem=60G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/imagepose/logs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/imagepose/logs/%j.err   # File to which STDERR will be written

SOURCE_DIR="/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf"
cd $SOURCE_DIR

# print info about current job
# echo $SLURM_JOB_ID
scontrol show job $SLURM_JOB_ID 

nvidia-smi

singularity exec --nv -B /mnt/qb/datasets,/mnt/qb/work/bethge/ahochlehnert48 docker://libeanim/ml-research:base /home/bethge/ahochlehnert48/.conda/envs/inerf/bin/python imagepose/train_model.py
