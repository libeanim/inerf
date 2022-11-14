import os

BASE_DATA_DIR = "../data/shapenet/chairs"
CONFIG_DIR = "./configs/chairs"

template = """expname = {name}
basedir = ./logs/chairs
datadir = {base_data_dir}/{name}
dataset_type = blender

half_res = True
no_batching = True

N_samples = 64
N_importance = 64

use_viewdirs = True

white_bkgd = True

N_rand = 1024
precrop_iters=500"""

chairs = os.listdir(BASE_DATA_DIR)

for chair in chairs:
    config = template.format(name=chair, base_data_dir=BASE_DATA_DIR)
    print('sbatch run.sbatch', os.path.join(CONFIG_DIR, f'{chair}.txt'))
    with open(os.path.join(CONFIG_DIR, f'{chair}.txt'), 'w') as f:
        f.write(config)
print(f"Done. Written {len(chairs)} files.")


#Fix dataset
# import json
# import os

# BASE_DATA_DIR = "../data/shapenet/chairs"
# chairs = os.listdir(BASE_DATA_DIR)


# def load_json(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     return data

# def save_json(data, path):
#     with open(path, 'w') as f:
#         data = json.dump(data, f, indent=2)
#     return data

# def fix_json(data):
#     for i, frame in enumerate(data['frames']):
#         frame['file_path'] = frame['file_path'].replace('.png', '')
#     return data

# filenames = ['transforms_train.json', 'transforms_val.json', 'transforms_test.json']

# for chair in chairs:
#     for filename in filenames:
#         path = os.path.join(BASE_DATA_DIR, chair, filename)
#         data = load_json(path)
#         data = fix_json(data)
#         save_json(data, path)
#         # break
#     # break

