import os.path as osp
import logging

import numpy
import numpy as np
import skimage.measure
import time
import torch
from torch.nn import functional as F
import os
import tqdm
import trimesh
import open3d as op

from Shape_GAN.fourier_feature import FourierFeatures
from Shape_GAN.datasets import PointDataset
from Shape_GAN.point_sdf_net import PointNet, SDFGenerator
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, scale_to_unit_sphere, BadMeshException


np.random.seed(0)
torch.manual_seed(0)

# checkpoint_path = '/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_shapeGAN_NDF_Sphere_abs/'
checkpoint_path = "/home/sebastian/ndf_gan/experiments/Pointnet_max_watertight_test/"
#checkpoint_path = "/home/sebastian/ndf_gan/experiments/Pointnet_Max_latent256_hidden512/"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

LATENT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True
N = 256

threshold = 0.5  # ich hatte bis jetzt 0.05

generator = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)

def load_checkpoint():
    path = checkpoint_path + 'checkpoint.tar'

    print('Loaded checkpoint from: {}'.format(path))
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['netG_state_dict'])
    epoch = checkpoint['epoch']
    config_counter = checkpoint['config_counter']
    return generator

generator = load_checkpoint()
generator = generator.to(device)
z = torch.randn(1, LATENT_SIZE, device=device)

for param in generator.parameters():
    param.requires_grad = False

array = np.asarray([[0, 0, 0], [0, 0.1, 0], [0, 0.11, 0],[0, 0.12, 0],[0, 0.13, 0],[0, 0.14, 0],[0, 0.15, 0],[0, 0.16, 0],[0, 0.17, 0],[0, 0.18, 0],[0, 0.19, 0],
                    [0, 0.2, 0],[0, 0.21, 0],[0, 0.22, 0],[0, 0.23, 0],[0, 0.24, 0],[0, 0.25, 0],[0, 0.26, 0],[0, 0.27, 0],[0, 0.28, 0],[0, 0.29, 0],
                    [0, 0.3, 0],[0, 0.31, 0],[0, 0.32, 0],[0, 0.33, 0],[0, 0.34, 0],
                    [0, 0.35, 0],[0, 0.36, 0],[0, 0.37, 0],[0, 0.38, 0],[0, 0.39, 0],
                    [0, 0.4, 0],[0, 0.41, 0],[0, 0.42, 0],[0, 0.43, 0], [0, 0.5, 0], [0, 0.6, 0], [0, 0.7, 0],[0, 0.8, 0], [0, 0.9, 0], [0, 1, 0]]).astype(np.float32)
samples = torch.from_numpy(array).unsqueeze(0).to(device)
samples.requires_grad = True

df_pred = generator(samples, z)

print("Finished")
