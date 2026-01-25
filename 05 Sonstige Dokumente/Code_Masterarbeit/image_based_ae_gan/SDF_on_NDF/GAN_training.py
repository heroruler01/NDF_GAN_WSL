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

import models.data.voxelized_data_shapenet as voxelized_data
import configs.config_loader as cfg_loader


cfg = cfg_loader.get_config()




class Trainer(object):

    def __init__(self, netG, netD,  device, val_dataset, exp_name, optimizerG, optimizerD, lr=1e-4, threshold=0.1):
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

        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints_GAN/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary_GAN'.format(exp_name))
        self.val_min = None
        self.max_dist = threshold
        self.grad_pen = 10



    def train_model(self, configuration):
        num_steps = 0
        device = self.device
        loss = 0
        start, training_time, configuration_start = self.load_checkpoint()
        iteration_start_time = time.time()

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

        for configuration_counter in range(configuration_start, 6):
            num_points, batch_size, epochs = configuration[configuration_counter]
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
            train_data_loader = train_dataset.get_loader()


            for epoch in range(0, epochs):
                sum_loss_D = 0
                sum_loss_G = 0
                counter = 0
                print('Start epoch {}'.format(epoch))


                for batch in train_data_loader:
                    num_steps += 1
                    #save model
                    iteration_duration = time.time() - iteration_start_time
                    if iteration_duration > 60 * 60:  # eve model every X min and at start
                        training_time += iteration_duration
                        iteration_start_time = time.time()
                        self.save_checkpoint(epoch, training_time, configuration_counter)

                    df_gt = batch.get('df').to(device)  # (Batch,num_points) #(4, 50000)
                    p = batch.get('grid_coords').to(device)  # grid coords are the sampled points with Sigma! not exactly at the surface

                    batch_counter = df_gt.size(0)

                    latent_vector = torch.randn(batch_counter, 128, device=device)
                    #latent_vector = torch.randn(batch_counter, 128, 50000, device=device)
                    # latent_vector = torch.randn(batch_counter, 3479, 50000, device=device)

                    self.optimizerD.zero_grad()

                    fake = self.netG(p, latent_vector)
                    out_real = self.netD(p, df_gt)  # df_gt and fake need to have the same shape
                    out_fake = self.netD(p, fake)
                    D_loss = out_fake.mean() - out_real.mean()  # in paper they say real - fake not vice versa!

                    # sampling ranodm values for the amount of points/distances building up mixed data: Gradient Penalty
                    # mixture of predicted and real values
                    alpha = torch.rand((batch_counter, p.size(1)), device=device)
                    fake = fake.squeeze(-1)
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
                    #torch.nn.utils.clip_grad_norm_(netD.parameters(), args.clip) --> before or after optimizer step

                    self.optimizerD.step()

                    # train the generator
                    if num_steps % 5 == 0:
                        self.optimizerG.zero_grad()

                        latent_vector = torch.randn(batch_counter, 128, device=device)
                        #latent_vector = torch.randn(batch_counter, 128, 50000, device=device)
                        # latent_vector = torch.randn(batch_counter, 3479, 50000, device=device)
                        fake = self.netG(p, latent_vector)
                        out_fake = self.netD(p, fake)
                        loss = -out_fake.mean()
                        loss.backward()
                        self.optimizerG.step()

                    # render image of the UDF
                    if num_steps % 100 == 0:

                        image = renderer.pred_render(self.netG, act_cam_or, act_cam_pos, self.device)


                    counter += 1

                    sum_loss_D += D_loss.abs().item()
                    sum_loss_G += loss

                    print('Epoch: {}, Loss Discriminator: {:.6f}, Loss Generator: {:.6f} progress (%): {}'.format(
                        epoch, loss_discriminator, loss, (counter / len(train_data_loader))*100))

                    # / self.train_dataset.num_sample_points

                    # print("Current loss: {}".format(loss / self.train_dataset.num_sample_points))


                self.writer.add_scalar('training loss Discriminator', sum_loss_D / len(train_data_loader), epoch)
                self.writer.add_scalar('training loss batch avg Generator', sum_loss_G / len(train_data_loader), epoch)




    def save_checkpoint(self, epoch, training_time, configuration_counter):
        path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(training_time),training_time])
        if not os.path.exists(path):
            torch.save({ #'state': torch.cuda.get_rng_state_all(),
                        'training_time': training_time ,'epoch': epoch, 'configuration_counter': configuration_counter,
                        'netG_state_dict': self.netG.state_dict(),
                        'optimizerG_state_dict': self.optimizerG.state_dict(),
                        'netD_state_dict': self.netD.state_dict(),
                        'optimizerD_state_dict': self.optimizerD.state_dict()}, path)



    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0, 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(*[*convertSecs(checkpoints[-1]),checkpoints[-1]])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)

        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        configuration_counter = checkpoint['configuration_counter']
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
        return epoch, training_time, configuration_counter


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds