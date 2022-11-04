import sys
sys.path.insert(0, '/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf')
import os
from tqdm import tqdm
import torch
import torchvision
from dataclasses import dataclass
from torch.utils.data import DataLoader
from datetime import datetime
from model import ImageToPosNet
from data import InMemoryDataset, PoseDataset

import numpy as np
from nerf_helpers import load_nerf
from render_helpers import render, to8b, get_rays
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt_id = datetime.now().isoformat()
print(f'ID: {dt_id}')


ROOT_DIR='/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/image-to-pos'
RESULT_DIR=f'{ROOT_DIR}/results'

## CONFIG
config = dict(
    id = dt_id,
    job_id = os.environ['SLURM_JOB_ID'],
    random_seed = 42,
    learning_rate = 0.001,
    # learning_rate = 0.5,
    # momentum = 0.9,
    epochs = 30,  # 18
    log_steps = 10,
    # weight_decay = 1e-4,
    weight_decay = 0,
    batch_size = 32,
    betas=(0.9, 0.999),
    optimizer='Adam'
)


torch.manual_seed(config["random_seed"])

@dataclass
class Args:
    netdepth=8
    netwidth=256
    netdepth_fine=8
    netwidth_fine=256
    chunk=1024*32
    netchunk=1024*64

    model_name='lego'
    data_dir = '/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/data/nerf_synthetic/'
    ckpt_dir = '/mnt/qb/work/bethge/ahochlehnert48/code/continual-nerf/inerf/ckpts'
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


default_nerf_args = Args()
render_kwargs = load_nerf(default_nerf_args, device)
bds_dict = {
    'near': 2.,
    'far': 6.5,
}
render_kwargs.update(bds_dict)

@torch.no_grad()
def render_image(pose):
    H, W = 400, 400
    camera_angle_x = test_dataset.camera_angle_x
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)

    rgb, disp, acc, extras = render(H, W, focal, chunk=default_nerf_args.chunk, rays=batch_rays,
                                    verbose=0 < 10, retraw=True,
                                    **render_kwargs)
    rgb = rgb.cpu().detach().numpy()
    rgb8 = to8b(rgb)
    return rgb8

@torch.no_grad()
def report_test_loss(total_step):
    running_loss = 0.
    plt.figure(figsize=(len(test_loader) * 4, 4*2))
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Test (#{total_step})'):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        gt_img = np.swapaxes(inputs[0].cpu().detach().numpy(), 0, 2)
        gt_img[:, :, 0] = gt_img[:, :, 0].T; gt_img[:, :, 1] = gt_img[:, :, 1].T; gt_img[:, :, 2] = gt_img[:, :, 2].T
        render_img = render_image(outputs[0])
        plt.subplot(2, len(test_loader), i + 1); plt.imshow(gt_img); plt.title(f'Ground truth (#{i * len(test_loader)})'); plt.axis('off')
        plt.subplot(2, len(test_loader), i + len(test_loader) + 1); plt.imshow(render_img); plt.title(f'Render (#{i * len(test_loader)})'); plt.axis('off')
        # wandb.log({
        #     'ground truth': wandb.Image(gt_img, mode='RGB', caption=f'Test Image #{i}'),
        #     'rendner': wandb.Image(render_img, mode='RGB', caption=f'Render Image #{i}')
        # }, step=total_step)
    wandb.log({
        'images': plt,
        'test_loss': running_loss / len(test_loader)}, step=total_step)
    return running_loss / len(test_loader)



import wandb
# wandb.init()
wandb.init(project='image-to-pos', entity='libeanim', name=dt_id, config=config)

# ds = PoseDataset('./image-to-pos/lego1', torchvision.transforms.ToTensor())
training_dataset = InMemoryDataset(f'{ROOT_DIR}/lego2.pickle')
training_loader = DataLoader(
    training_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=2, 
)

test_dataset = PoseDataset('./image-to-pos/lego-test', torchvision.transforms.ToTensor())
test_loader = DataLoader(
    test_dataset, 
    batch_size=config['batch_size'],
    shuffle=False, 
    num_workers=4,
)


torch.set_default_tensor_type('torch.cuda.FloatTensor')
model = ImageToPosNet(3)

wandb.watch(model)

model.cuda()

loss_fn = torch.nn.MSELoss()

# Optimizers specified in the torch.optim package
if config['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
elif config['optimizer'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"],
                                weight_decay=config["weight_decay"])
else:
    raise ValueError('Optimizer not set correctly.')

for epoch_index in range(config["epochs"]):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(training_loader), total=len(training_loader), desc=f'Train (#{epoch_index})'):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # print('TARGET', labels)
        # print('OUTPUT', outputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % config["log_steps"] == config["log_steps"] - 1:
            # test_accuracy = get_accuracy(model, val_loader)
            total_step = epoch_index * len(training_loader) + i
            wandb.log({
                "loss": running_loss / config["log_steps"],
                "lr": optimizer.param_groups[0]['lr'],
                "batch": i,
                "epoch": epoch_index
            }, step=total_step)
            running_loss = 0.0
        # if i % 100 == 99:
        #     last_loss = running_loss / 10 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.
    report_test_loss((epoch_index + 1) * len(training_loader))
    # test_loss = 0.
    # wandb.log({"test_loss": test_loss}, step=epoch_index * len(training_loader) + i)
    
    # print('Last Epoch: {} - Loss: {}'.format(epoch_index, running_loss / i))

torch.save(model.state_dict(), f"{RESULT_DIR}/{ config['id'] }_final.pth")
print('Model saved.')
