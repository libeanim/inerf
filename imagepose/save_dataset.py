
import torchvision
from data import PoseDataset, InMemoryDataset

DATASET_PATH = '/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/image-to-pos/lego2'

ds = PoseDataset(DATASET_PATH, torchvision.transforms.ToTensor())
InMemoryDataset.create_dataset_file(ds, f'{DATASET_PATH}.pickle')

# srun --pty --mem=50G --partition=cpu-short -c 2 singularity exec -B /scratch_local,/mnt/qb/work/bethge/ahochlehnert48 docker://libeanim/ml-research:base /home/bethge/ahochlehnert48/.conda/envs/inerf/bin/python imagepose/save_dataset.py