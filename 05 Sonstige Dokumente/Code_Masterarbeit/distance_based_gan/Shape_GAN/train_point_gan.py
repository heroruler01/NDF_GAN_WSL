import os.path as osp
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import RMSprop
import random
from datasets import PointDataset
from point_sdf_net import PointNet, SDFGenerator

from torch.utils.tensorboard import SummaryWriter
import open3d as op

from fourier_feature import FourierFeatures

#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True) # shapenet/cars_internal_preprocessed
args = parser.parse_args()
writer = SummaryWriter(log_dir="runs/pointnet_mix_fourier_selected1000") # "runs" folder in the ShapeGAN directory

checkpoint_path = "/home/sebastian/ndf_gan/experiments/pointnet_mix_fourier_selected1000/"
if not os.path.exists(checkpoint_path):
    print(checkpoint_path)
    os.makedirs(checkpoint_path)
config_counter = 0

LATENT_SIZE = 128 # standard 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256 # standard 256
NUM_LAYERS = 8 # standard 8
NORM = True

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(G))
print(count_parameters(D))

#G= torch.nn.DataParallel(G,device_ids = [0, 1])
#D= torch.nn.DataParallel(D,device_ids = [0, 1])
G, D = G.to(device), D.to(device)
G_optimizer = RMSprop(G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

root = osp.join(f'/home/sebastian/ndf_gan/Shape_GAN/data/{args.category}')
dataset = PointDataset.from_split(root, split='train')


configuration = [  # num_points, batch_size, epochs
    (1024, 32, 300),
    (2048, 32, 300),
    (4096, 32, 300),
    (8192, 24, 300),
    (16384, 12, 300),
    (32768, 12, 900),
]
'''

# config 16normal
configuration = [  # num_points, batch_size, epochs
    (1024, 32, 300),
    (2048, 32, 300),
    (4096, 32, 300),
    (8192, 24, 300),
    (16384, 12, 500),
]


# config16
configuration = [  # num_points, batch_size, epochs
    (32768, 12, 1200),
]
'''
config_index = 0
num_steps = 0
# ---------------------------------------
# arguments for fourier features
# ---------------------------------------

num_frequencies = 128 #standard is 128
std_dev = 1
input_dim = 3  # das sind die Anzahl an channels (x,y,z)

if num_frequencies:
    frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                    std=std_dev).to(device) # einfach eine Matrix (128,3)
    encoding = FourierFeatures(frequency_matrix)
else:
    encoding = torch.nn.Identity()

# ---------------------------------------

for num_points, batch_size, epochs in configuration:
    dataset.num_points = num_points
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=6)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_loss_G = 0
        for uniform in loader:
            num_steps += 1

            uniform = uniform.to(device)
            u_pos, u_dist = uniform[..., :3], uniform[..., 3:]  # Tensor:(batch size, points, 3)

            '''
            pcd = op.geometry.PointCloud()
            pcd.points = op.utility.Vector3dVector(u_pos[1].detach().cpu().numpy())
            op.visualization.draw_geometries([pcd])
            '''
            # change SDF values to NDF by taking the abs() of the distances
            u_dist = torch.abs(u_dist)


            # ---------------------------------------------------
            '''Code for Fourier Feature Trafo of u_pos
            In order to use these features, we have to change u_pos to f_features everywhere it is used.
            We have to change size of the input layer from 4 (3+1) to 257 (128*2 frequencies +1 (distance))
            '''
            coordinates = u_pos
            f_features = encoding(coordinates)

            # ---------------------------------------------------


            D_optimizer.zero_grad()

            z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
            fake = G(u_pos, z)
            out_real = D(f_features, u_dist)
            out_fake = D(f_features, fake)
            D_loss = out_fake.mean() - out_real.mean()

            alpha = torch.rand((uniform.size(0), 1, 1), device=device)
            interpolated = alpha * u_dist + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            out = D(f_features, interpolated)

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
                z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
                fake = G(u_pos, z)
                out_fake = D(f_features, fake)
                loss = -out_fake.mean()
                loss.backward()
                G_optimizer.step()

            total_loss += D_loss.abs().item()
            total_loss_G += loss

        print('Num points: {}, Epoch: {:03d}, Loss: {:.6f}'.format(
            num_points, epoch, total_loss / len(loader)))

        config_index += 1 # zählt Gesamtanzahl an Epochen über configurations

        writer.add_scalar('training_loss_Discriminator', total_loss / len(loader), config_index)
        writer.add_scalar('training_loss_batch_avg_Generator', total_loss_G / len(loader), config_index)

        path = checkpoint_path + 'checkpoint.tar'
        torch.save({  # 'state': torch.cuda.get_rng_state_all(),
            'epoch': epoch, 'config_counter': config_counter, 'frequency_matrix': frequency_matrix,
            'netG_state_dict': G.state_dict(),
            'optimizerG_state_dict': G_optimizer.state_dict(),
            'netD_state_dict': D.state_dict(),
            'optimizerD_state_dict': D_optimizer.state_dict()}, path)

    config_counter +=1 # zählt Anzahl an Epochen und wird bei jeder Configurations zurückgesetzt

print('Finished Training successfully')
