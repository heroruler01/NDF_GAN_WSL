from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time

import GAN.GAN_renderer as renderer

from torch.utils.data import DataLoader




class Trainer(object):

    def __init__(self, netG, netD,  device, train_dataset, val_dataset, exp_name, optimizerG, optimizerD, lr = 1e-4, threshold = 0.1):
        self.netG = netG.to(device)
        self.netD = netD.to(device)
        self.device = device
        if optimizerG == 'Adam':
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr)
        if optimizerG == 'Adadelta':
            self.optimizerG = optim.Adadelta(self.netG.parameters())
        if optimizerG == 'RMSprop':
            self.optimizerG = optim.RMSprop(self.netG.parameters(), momentum=0.9)

        if optimizerD == 'Adam':
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr)
        if optimizerD == 'Adadelta':
            self.optimizerD = optim.Adadelta(self.netD.parameters())
        if optimizerD == 'RMSprop':
            self.optimizerD = optim.RMSprop(self.netD.parameters(), momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)

        self.writer = SummaryWriter(self.exp_path + 'summary_GAN_RMS'.format(exp_name))
        self.val_min = None
        self.max_dist = threshold
        self.grad_pen = 10


    def train_model(self, epochs):
        checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_GAN_airplanes_RMS/"
        if not os.path.exists(checkpoint_path):
            print(checkpoint_path)
            os.makedirs(checkpoint_path)

        num_steps = 0
        device = self.device
        loss = 0


        ndf_cam_position = [[1.5, 0, 0], [0, -1.5, 0], [0, 1.5, 0], [-1.5, 0, 0], [0, 0, 1.5], [0, 0, -1.5]]
        ndf_cam_orientation = [[90.0, 0.0, 90.0], [90.0, 0.0, 0.0], [90.0, 0.0, 180.0], [90.0, 0.0, 270.0],
                               [0.0, 0.0, 0.0], [180.0, 0.0, 0]]
        act_cam_pos = ndf_cam_position[2]
        act_cam_or = ndf_cam_orientation[2]

        configuration = [  # num_points, batch_size, epochs
            (1024, 32, 300),
            (2048, 32, 300),
            (4096, 32, 300),
            (8192, 24, 300),
            (16384, 12, 300),
            (32768, 6, 900),
        ]

        config_index = 0
        config_counter = 0

        num_steps = 0
        for num_points, batch_size, epochs in configuration:
            self.train_dataset.num_points = num_points
            loader = DataLoader(self.train_dataset, batch_size, shuffle=True, num_workers=6)

            for epoch in range(1, epochs + 1):
                total_loss = 0
                total_loss_G = 0
                counter = 0
                print('Start epoch {}'.format(epoch))


                for uniform, _ in loader:
                    num_steps += 1
                    uniform = uniform.to(device)
                    p, df_gt = uniform[..., :3], uniform[..., 3:]

                    #df_gt = batch.get('df').to(device)  # (Batch,num_points) #(4, 50000)
                    #p = batch.get('grid_coords').to(device)  # grid coords are the sampled points with Sigma! not exactly at the surface

                    batch_counter = df_gt.size(0)

                    latent_vector = torch.randn(uniform.size(0), 128, num_points, device=device)
                    #latent_vector = torch.randn(batch_counter, 128, 50000, device=device)
                    # latent_vector = torch.randn(batch_counter, 3479, 50000, device=device)

                    self.optimizerD.zero_grad()

                    fake = self.netG(p, latent_vector)
                    out_real = self.netD(p, df_gt)  # df_gt and fake need to have the same shape
                    out_fake = self.netD(p, fake)
                    D_loss = out_fake.mean() - out_real.mean()  # in paper they say real - fake not vice versa!

                    # sampling ranodm values for the amount of points/distances building up mixed data: Gradient Penalty
                    # mixture of predicted and real values
                    alpha = torch.rand((uniform.size(0), 1, 1), device=device)
                    fake = fake.unsqueeze(-1)
                    interpolated = alpha * df_gt + (1 - alpha) * fake
                    interpolated.requires_grad_(True)
                    out = self.netD(p, interpolated)
                    grad = torch.autograd.grad(out, interpolated,
                                               grad_outputs=torch.ones_like(out),
                                               create_graph=True, retain_graph=True,
                                               only_inputs=True)[0]
                    grad_norm = grad.view(grad.size(0), -1).norm(dim=-1, p=2)
                    gp = self.grad_pen * ((grad_norm - 1).pow(2).mean())

                    loss_discriminator = D_loss + gp
                    loss_discriminator.backward()

                    # should i clip gradients of the discriminator? --> no, gradient penalty enforces Lipschitz Constraint

                    self.optimizerD.step()

                    # train the generator
                    if num_steps % 5 == 0:
                        self.optimizerG.zero_grad()

                        latent_vector = torch.randn(uniform.size(0), 128, num_points, device=device)
                        #latent_vector = torch.randn(batch_counter, 128, 50000, device=device)
                        # latent_vector = torch.randn(batch_counter, 3479, 50000, device=device)
                        fake = self.netG(p, latent_vector)
                        out_fake = self.netD(p, fake)
                        loss = -out_fake.mean()
                        loss.backward()
                        self.optimizerG.step()

                    total_loss += D_loss.abs().item()
                    total_loss_G += loss
                    # render image of the UDF





                    total_loss += D_loss.abs().item()
                    total_loss_G += loss

                print('Num points: {}, Epoch: {:03d}, Loss: {:.6f}'.format(
                    num_points, epoch, total_loss / len(loader)))

                config_index += 1

                #if num_steps % 100 == 0:
                 #   image = renderer.pred_render(self.netG, act_cam_or, act_cam_pos, self.device)

                self.writer.add_scalar('training loss Discriminator', total_loss / len(loader), config_index)
                self.writer.add_scalar('training loss batch avg Generator', total_loss_G / len(loader), config_index)

                path = checkpoint_path + 'checkpoint.tar'
                torch.save({  # 'state': torch.cuda.get_rng_state_all(),
                    'epoch': epoch, 'config_counter': config_counter,
                    'netG_state_dict': self.netG.state_dict(),
                    'optimizerG_state_dict': self.optimizerG.state_dict(),
                    'netD_state_dict': self.netD.state_dict(),
                    'optimizerD_state_dict': self.optimizerD.state_dict()}, path)

            config_counter += 1

        print('Finished Training successfully')
