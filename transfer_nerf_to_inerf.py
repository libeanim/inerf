import os
from glob import glob
import re

LOAD_DIR = './nerf/logs/chairs'
SAVE_DIR = './inerf/ckpts/chairs'

def get_file_names(path: str):
    COARSE_NAME = 'model_*.npy'
    FINE_NAME = 'model_fine*.npy'

    model_coarse = get_newest_checkpoint(path, COARSE_NAME)
    model_fine = get_newest_checkpoint(path, FINE_NAME)

    return model_coarse, model_fine
    

def get_newest_checkpoint(path: str, file_name: str):
    search_pattern = os.path.join(path, file_name)
    files = glob(search_pattern)
    iterations = list(map(get_number, files))
    max_id = [i for i, j in enumerate(iterations) if j == max(iterations)][0]
    return files[max_id]

def get_number(file_name: str) -> int:
    base_name = os.path.basename(file_name)
    regex = re.compile('model(_fine){0,1}_([0-9]*)\.npy')
    m = regex.match(base_name)
    return int(m.groups()[1])


chairs = os.listdir(LOAD_DIR)

for chair in chairs:
    if chair == 'summaries':
        continue
    chair_path = os.path.join(LOAD_DIR, chair)
    coarse, fine = get_file_names(chair_path)
    coarse_target = os.path.join(SAVE_DIR, f'{chair}.npy')
    fine_target = os.path.join(SAVE_DIR, f'{chair}_fine.npy')

    # print(f'cp {coarse} {coarse_target}')
    # print(f'cp {fine} {fine_target}')

    os.system(f'cp {coarse} {coarse_target}')
    os.system(f'cp {fine} {fine_target}')
