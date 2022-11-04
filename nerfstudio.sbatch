#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu-2080ti
#SBATCH --time=3-00:00
#SBATCH --mem=60G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1  # gpu:2
#SBATCH --output=/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/outputs/logs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/outputs/logs/%j.err   # File to which STDERR will be written

SOURCE_DIR="/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf"
cd $SOURCE_DIR


# print info about current job
# echo $SLURM_JOB_ID
scontrol show job $SLURM_JOB_ID 

nvidia-smi

SHAPENET_MODEL='chair_50'
singularity exec --nv \
    -B /mnt/qb/datasets,/mnt/qb/work/bethge/ahochlehnert48 docker://libeanim/ml-research:code-blender \
    /home/bethge/ahochlehnert48/.conda/envs/nerfstudio/bin/ns-train vanilla-nerf --vis wandb --machine.num-gpus 1 \
    --trainer.max-num-iterations 100000 --pipeline.model.collider-params near_plane 0.5 far_plane 9.0 --pipeline.model.num-importance-samples 128 \
    blender-data --data ./data/shapenet/$SHAPENET_MODEL

# singularity exec --nv \
#     -B /mnt/qb/datasets,/mnt/qb/work/bethge/ahochlehnert48 docker://libeanim/ml-research:code-blender \
#     /home/bethge/ahochlehnert48/.conda/envs/nerfstudio/bin/ns-train vanilla-nerf --vis wandb --machine.num-gpus 1 blender-data --data ./data/nerf_synthetic/chair