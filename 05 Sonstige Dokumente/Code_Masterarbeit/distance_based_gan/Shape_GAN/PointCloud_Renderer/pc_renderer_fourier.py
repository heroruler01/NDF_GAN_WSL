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
#np.random.seed(0)
#torch.manual_seed(0)
# checkpoint_path = '/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_shapeGAN_NDF_Sphere_abs/'
checkpoint_path = "/home/sebastian/ndf_gan/experiments/Pointnet_max_fourier_G/"
#checkpoint_path = "/home/sebastian/ndf_gan/experiments/Pointnet_Max_latent256_hidden512/"
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

LATENT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True
N = 256
FOURIER = True
threshold = 0.1  # ich hatte bis jetzt 0.05

generator = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)


def load_checkpoint():
    '''
    checkpoints = glob(checkpoint_path + '/*')
    if len(checkpoints) == 0:
        print('No checkpoints found at {}'.format(checkpoint_path))
        return 0, 0
    '''
    path = checkpoint_path + 'checkpoint.tar'

    print('Loaded checkpoint from: {}'.format(path))
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['netG_state_dict'])
    epoch = checkpoint['epoch']
    config_counter = checkpoint['config_counter']
    frequency_matrix = checkpoint['frequency_matrix']
    return generator, frequency_matrix

generator, frequency_matrix = load_checkpoint()
generator = generator.to(device)
z = torch.randn(1, LATENT_SIZE, device=device)


if FOURIER:
    encoding = FourierFeatures(frequency_matrix)

def generate_point_cloud(num_steps=10, num_points=900000, filter_val=0.02):#0.009, good is 0.02 in combination with 0.06 below
    start = time.time()
    # inputs = data['inputs'].to(self.device)

    for param in generator.parameters():
        param.requires_grad = False

    sample_num = 200000
    samples_cpu = np.zeros((0, 3))

    unit_sphere_points = np.random.uniform(-1, 1, size=(sample_num * 2, 3)).astype(np.float32)
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]
    samples = torch.from_numpy(unit_sphere_points[:sample_num, :]).unsqueeze(0).to(device)

    sample_fourier = encoding(samples)

    # below samples in a 1 square
    #samples = torch.from_numpy(np.random.uniform(-1, 1, size=(1, sample_num, 3)).astype(np.float32)).to(device) # da muss noch eine 1 davor

    # below samples in a bigger square
    #samples = torch.rand(1, sample_num, 3).float().to(device) * 3 - 1.5

    # Code below is a visualization of how the points are sampled initially!
    '''
    samples_pc = samples.squeeze()
    samples_pc = samples_pc.detach().cpu().numpy()
    pcd = op.geometry.PointCloud()
    pcd.points = op.utility.Vector3dVector(samples_pc)
    op.visualization.draw_geometries([pcd])
    '''

    samples.requires_grad = True
    sample_fourier.requires_grad = True

    i = 0
    while len(samples_cpu) < num_points:
        print('iteration', i)

        for j in range(num_steps):
            print('refinement', j)
            df_pred = torch.clamp(generator(sample_fourier, z), max=threshold).squeeze(0)
            df_pred = df_pred.transpose(1,0)
            df_pred.sum().backward()

            gradient = sample_fourier.grad.detach()
            samples = samples.detach()
            df_pred = df_pred.detach()
            # inputs = inputs.detach()

            samples = samples - F.normalize(gradient, dim=2) * df_pred.reshape(-1, 1)  # better use Tensor.copy method?
            samples = samples.detach()
            samples.requires_grad = True

        print('finished refinement')

        if not i == 0:
            samples_cpu = np.vstack((samples_cpu, samples[df_pred < filter_val].detach().cpu().numpy()))

        #tes = df_pred.detach().cpu().numpy()
        #tes = np.squeeze(tes)
        #tes = tes < 0.03
        #idx = np.argwhere(np.asarray(tes))
        #samples = samples[idx].unsqueeze(0)
        samples = samples[df_pred < 0.06]#.unsqueeze(0)#0.03


        samples = samples[torch.norm(samples, p="fro", dim=1) < 1].unsqueeze(0)

        '''
        samples_pc = samples.squeeze()
        samples_pc = samples_pc.detach().cpu().numpy()
        pcd = op.geometry.PointCloud()
        pcd.points = op.utility.Vector3dVector(samples_pc)
        op.visualization.draw_geometries([pcd])
        '''

        indices = torch.randint(samples.shape[1], (1, sample_num))
        samples = samples[[[0, ] * sample_num], indices]
        samples += (threshold / 3) * torch.randn(samples.shape).to(device)  # 3 sigma rule
        samples = samples.detach()
        samples.requires_grad = True

        i += 1
        print(samples_cpu.shape)

    duration = time.time() - start

    return samples_cpu, duration


out_path = '/home/sebastian/ndf_gan/Shape_GAN/plots/'

if not os.path.exists(out_path):
    os.makedirs(out_path)
print(out_path)

export_path = out_path

# generate_point_cloud(self, num_steps=10, num_points=900000, filter_val=0.009)
point_cloud, duration = generate_point_cloud()
point_cloud = point_cloud[np.linalg.norm(point_cloud, axis=1) < 1]
pcd = op.geometry.PointCloud()
pcd.points = op.utility.Vector3dVector(point_cloud)
#op.visualization.draw_geometries([pcd])

op.io.write_point_cloud("/home/sebastian/ndf_gan/Shape_GAN/plots/dense_point_cloud.ply", pcd)

np.savez(export_path + 'dense_point_cloud', point_cloud=point_cloud, duration=duration)
trimesh.Trimesh(vertices=point_cloud, faces=[]).export(
    export_path + 'dense_point_cloud_trimesh.off')

