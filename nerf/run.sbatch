#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-2080ti
#SBATCH --time=3-00:00
#SBATCH --mem=30G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1  # gpu:2
#SBATCH --output=/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/nerf/logs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/nerf/logs/%j.err   # File to which STDERR will be written

SOURCE_DIR="/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/nerf"
cd $SOURCE_DIR


# print info about current job
# echo $SLURM_JOB_ID
scontrol show job $SLURM_JOB_ID 

nvidia-smi

# SHAPENET_MODEL='chair_40_1'
# singularity exec --nv \
#     -B /mnt/qb/datasets,/mnt/qb/work/bethge/ahochlehnert48 docker://libeanim/ml-research:code-blender \
#     /home/bethge/ahochlehnert48/.conda/envs/nerfstudio/bin/ns-train vanilla-nerf --vis wandb --machine.num-gpus 1 blender-data --data ./data/shapenet/$SHAPENET_MODEL

# CONFIG='configs/config_chair.txt'
# CONFIG='configs/config_chair40.txt'
# CONFIG='configs/config_chair40_1.txt'
CONFIG='configs/config_chair50.txt'
echo "Using config: $CONFIG"
singularity exec --nv \
    -B /mnt/qb/datasets,/mnt/qb/work/bethge/ahochlehnert48 docker://libeanim/ml-research:code-blender \
    /home/bethge/ahochlehnert48/.conda/envs/nerf/bin/python run_nerf.py --config $CONFIG