import os.path as osp
import os
import time
import argparse
import torch
import numpy as np
from glob import glob
# from torch.utils.data import DataLoader
from torch.optim import RMSprop

# from datasets import PointDataset
from SDF_on_NDF.original_SDF_GAN_architecture import PointNet, SDFGenerator
from torch.utils.tensorboard import SummaryWriter
import models.data.voxelized_data_shapenet as voxelized_data
import configs.config_loader as cfg_loader

import GAN.GAN_renderer as renderer

cfg = cfg_loader.get_config()

# parser = argparse.ArgumentParser()
# parser.add_argument('--category', type=str, required=True)
# args = parser.parse_args()

LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

exp_name = cfg.exp_name
threshold = 0.1

val_min = None
max_dist = threshold
grad_pen = 10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)
G, D = G.to(device), D.to(device)
G_optimizer = RMSprop(G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

# root = osp.join(f'data/{args.category}')
# dataset = PointDataset.from_split(root, split='train')


checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_shapegan_NDFclose/"
if not os.path.exists(checkpoint_path):
    print(checkpoint_path)
    os.makedirs(checkpoint_path)

writer = SummaryWriter(log_dir="runs/shapeganndfclose") # exp_path + 'summary_GAN_TEST'.format(exp_name)


configuration = [  # num_points, batch_size, epochs
    (1024, 32, 300),
    (2048, 32, 300),
    (4096, 32, 300),
    (8192, 24, 300),
    (16384, 12, 300),
    (32768, 6, 900),
]

num_steps = 0
config_index = 0
config_counter = 0
loss = 0

ndf_cam_position = [[1.5, 0, 0], [0, -1.5, 0], [0, 1.5, 0], [-1.5, 0, 0], [0, 0, 1.5], [0, 0, -1.5]]
ndf_cam_orientation = [[90.0, 0.0, 90.0], [90.0, 0.0, 0.0], [90.0, 0.0, 180.0], [90.0, 0.0, 270.0],
                       [0.0, 0.0, 0.0], [180.0, 0.0, 0]]
act_cam_pos = ndf_cam_position[2]
act_cam_or = ndf_cam_orientation[2]

for num_points, batch_size, epochs in configuration:
    # dataset.num_points = num_points
    # loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=6)
    train_dataset = voxelized_data.VoxelizedDataset('train',
                                                    res=cfg.input_res,
                                                    pointcloud_samples=cfg.num_points,
                                                    data_path=cfg.data_dir,
                                                    split_file=cfg.split_file,
                                                    batch_size=batch_size,
                                                    num_sample_points=num_points,
                                                    num_workers=30,
                                                    sample_distribution=cfg.sample_ratio,
                                                    sample_sigmas=cfg.sample_std_dev)
    loader = train_dataset.get_loader()


    for epoch in range(epochs):
        total_loss_D = 0
        total_loss_G = 0
        G_loss = 0
        print('Start epoch {}'.format(epoch))

        for batch in loader:
            num_steps += 1

            df_gt = batch.get('df').to(device)  # (Batch,num_points) #(4, 50000)
            p = batch.get('grid_coords').to(device)  # grid coords are the sampled points with Sigma! not exactly at the surface

            D_optimizer.zero_grad()

            batch_counter = df_gt.size(0)

            z = torch.randn(batch_counter, 128, device=device)
            fake = G(p, z)
            fake = fake.squeeze(-1)
            out_real = D(p, df_gt)
            out_fake = D(p, fake)
            D_loss = out_fake.mean() - out_real.mean()

            alpha = torch.rand((batch_counter, 1), device=device)

            interpolated = alpha * df_gt + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            out = D(p, interpolated)

            grad = torch.autograd.grad(out, interpolated,
                                       grad_outputs=torch.ones_like(out),
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(dim=-1, p=2)
            gp = GRADIENT_PENALITY * ((grad_norm - 1).pow(2).mean())

            loss = D_loss + gp
            loss.backward()
            D_optimizer.step()

            if num_steps % 5 == 0:
                G_optimizer.zero_grad()
                z = torch.randn(batch_counter, 128, device=device)
                fake = G(p, z)
                out_fake = D(p, fake)
                G_loss = -out_fake.mean()
                G_loss.backward()
                G_optimizer.step()

            total_loss_D += D_loss.abs().item()
            total_loss_G += G_loss

        print('Num points: {}, Epoch: {:03d}, Loss_D: {:.6f}'.format(
            num_points, epoch, total_loss_D / len(loader)))

        # render image of the UDF
        image = renderer.pred_render(G, act_cam_or, act_cam_pos, device)



        writer.add_scalar('training loss Discriminator', total_loss_D / len(loader), config_index)
        writer.add_scalar('training loss batch avg Generator', total_loss_G / len(loader), config_index)

        config_index += 1

        path = checkpoint_path + 'checkpoint.tar'
        torch.save({  # 'state': torch.cuda.get_rng_state_all(),
            'epoch': epoch, 'config_counter': config_counter,
            'netG_state_dict': G.state_dict(),
            'optimizerG_state_dict': G_optimizer.state_dict(),
            'netD_state_dict': D.state_dict(),
            'optimizerD_state_dict': D_optimizer.state_dict()}, path)

    config_counter += 1

print('Finished Training successfully')