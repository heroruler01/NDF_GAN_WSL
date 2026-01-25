import os.path as osp
import os
import argparse
import torch
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from datasets import PointDataset
from point_sdf_net import PointNet, SDFGenerator
import GAN.GAN_renderer as renderer

ndf_cam_position = [[1.5, 0, 0], [0, -1.5, 0.5], [0, 1.5, 0], [-1.5, 0, 0], [0, 0, 1.5], [0, 0, -1.5]]
ndf_cam_orientation = [[90.0, 0.0, 90.0], [90.0, 0.0, 0.0], [90.0, 0.0, 180.0], [90.0, 0.0, 270.0],
                       [0.0, 0.0, 0.0], [180.0, 0.0, 0]]
act_cam_pos = ndf_cam_position[1]
act_cam_or = ndf_cam_orientation[1]

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True
#checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_shapeGAN/"
checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoint_shapegan_ABS/"
def load_checkpoint():
    checkpoints = glob(checkpoint_path + '/*')
    if len(checkpoints) == 0:
        print('No checkpoints found at {}'.format(checkpoint_path))
        return 0, 0

    path = checkpoint_path + 'checkpoint.tar'

    print('Loaded checkpoint from: {}'.format(path))
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['netG_state_dict'])
    epoch = checkpoint['epoch']
    config_counter = checkpoint['config_counter']
    # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
    return epoch, config_counter, G



G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)
G, D = G.to(device), D.to(device)

epoch, config_counter, G = load_checkpoint()
print(epoch)
image = renderer.pred_render(G, act_cam_or, act_cam_pos, device)
