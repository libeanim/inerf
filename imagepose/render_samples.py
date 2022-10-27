import sys
sys.path.insert(0, '/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf')
import typing
import torch
import numpy as np
import numpy.typing as npt

from nerf_helpers import load_nerf
from render_helpers import render, to8b, get_rays
from scipy.spatial.transform import Rotation as R
import imageio
from tqdm import tqdm

import os
import json
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataclasses import dataclass

@dataclass
class Args:
    netdepth=8
    netwidth=256
    netdepth_fine=8
    netwidth_fine=256
    chunk=1024*32
    netchunk=1024*64

    model_name='lego'
    data_dir = './data/nerf_synthetic/'
    ckpt_dir = './ckpts'
    dataset_type='blender'
    obs_img_num=1
    
    use_viewdirs=True
    N_importance=64
    N_samples=64
    white_bkgd=True
    
    i_embed=0
    perturb=0.
    raw_noise_std=0.
    lindisp=True
    multires=10
    multires_views=4

    sample_count=10_000
    # sample_count=250


args = Args()

testsavedir = './image-to-pos/lego2'


# random_state = np.load(os.path.join(testsavedir, 'random_state.npy'), allow_pickle=True).tolist()
# np.random.set_state(random_state)
# start_index = 872
start_index=0
np.save(os.path.join(testsavedir, 'random_state.npy'), np.asanyarray(np.random.get_state(), dtype=object))



trans_t = lambda t: np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1]])

def sample_from_sphere_uniform(ndim: int, r: float) -> npt.NDArray[typing.Any]:
    """Samples a random point on the `ndim`-dimensional sphere of radius `r`.
    
    Exploits that a multivariate normal distribution is spherical in $\mathbb R^n$, then normalizes and scales by `r`.
    """
    out = np.zeros(ndim)
    while np.linalg.norm(out) == 0:
        out = np.random.randn(ndim)

    out /= np.linalg.norm(out)
    out *= r
    return out


def get_random_pose(d_min: float=3, d_max: float=5, t_min: float=0, t_max: float=0.5, clamp_uh: bool=True) -> npt.NDArray[typing.Any]:
    """Returns the transformation matrix (in homogeneous coordinates) for a random 6D starting pose.
    The pose is defined by a 3D rotation and a 3D translation.
    The rotation is sampled uniformly on SO(3) or its subset which results in a viewpoint in the upper hemisphere if `clamp_uh` is True.
    The translation along the z-axis (view axis, equivalent to distance from object) is sampled uniformly from [`d_min`, `d_max`].
    Translations along the x- and y-axis (orthogonal to view axis) are sampled uniformly from [`t_min`, `t_max`].
    """
    # warnings.warn("Don't forget to seed numpy for reproducability when using this function.")
    # sample uniform rotation
    rotation = np.eye(4)
    rotation[:3, :3] = R.random().as_matrix()
    while clamp_uh and np.dot([0, 0, 1], np.dot(rotation[:3, :3], [0, 0, 1])) < 0:
        rotation[:3, :3] = R.random().as_matrix()

    # sample translation
    z = np.random.uniform(d_min, d_max, 1)
    xy = np.random.uniform(t_min, t_max, 2)
    translation = trans_t(np.concatenate([xy, z]))

    # rotate around origin, then translate
    pose = rotation @ translation
    return torch.Tensor(pose)


torch.set_default_tensor_type('torch.cuda.FloatTensor')
H, W = 400, 400
camera_angle_x = 0.6911112070083618
focal = .5 * W / np.tan(.5 * camera_angle_x)

with open(os.path.join(args.data_dir + str(args.model_name) + "/obs_imgs/", 'transforms.json'), 'r') as fp:
    meta = json.load(fp)
frames = meta['frames']

obs_img_pose = np.array(frames[args.obs_img_num]['transform_matrix']).astype(np.float32)

d = np.linalg.norm(obs_img_pose[:3, 3:])

delta_d = (-1, 2)

render_kwargs = load_nerf(args, device)
bds_dict = {
    'near': 2.,
    'far': 6.5,
}
render_kwargs.update(bds_dict)

# poses = [ get_random_pose(d+delta_d[0], d+delta_d[1], 0., 1., clamp_uh=True).cuda() for _ in range(start_index, 10_000) ]

frames = []
for i in range(start_index, args.sample_count):
    pose = get_random_pose(d+delta_d[0], d+delta_d[1], 0., 1., clamp_uh=True).cuda()
    frames.append({
        "file_path": str(i),
        "transform_matrix": pose
    })

print('Save poses')
with open(os.path.join(testsavedir, 'transforms.json'), 'w') as f:
    # pc = [p.cpu().numpy().tolist() for p in poses]
    tmp = []
    for frame in frames:
        tmp.append({
            "file_path": frame['file_path'],
            "transform_matrix": frame['transform_matrix'].cpu().numpy().tolist()
        })
    json.dump({
        "camera_angle_x": camera_angle_x,
        "frames": tmp
    }, f, indent=2)

# plt.figure(figsize=(20, 10))
print(f'Render {len(frames)} images')
for frame in tqdm(frames):
    pose = frame['transform_matrix']

    rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)

    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                    verbose=0 < 10, retraw=True,
                                    **render_kwargs)
    rgb = rgb.cpu().detach().numpy()
    rgb8 = to8b(rgb)
    filename = os.path.join(testsavedir, f'{frame["file_path"]}.png')
    imageio.imwrite(filename, rgb8)

#     plt.subplot(2, 5, i+1)
#     plt.imshow(rgb)

# plt.show()



## srun --pty --mem=40G --gres=gpu:1 --partition=gpu-2080ti -c 4 singularity exec -B /scratch_local,/mnt/qb/work/bethge/ahochlehnert48 --nv docker://libeanim/ml-research:base /home/bethge/ahochlehnert48/.conda/envs/inerf/bin/python render_samples.py