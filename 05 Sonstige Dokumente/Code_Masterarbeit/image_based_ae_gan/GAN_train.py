import GAN.GAN as model
#import GAN.GAN as model
import models.data.voxelized_data_shapenet as voxelized_data
from GAN import GAN_training_airplanes
import torch
import configs.config_loader as cfg_loader
from Shape_GAN.datasets import PointDataset

cfg = cfg_loader.get_config()
netG = model.NDF()
netD = model.Discriminator()


# -------------------------
# Standard NDF Dataset
# -------------------------
'''
train_dataset = voxelized_data.VoxelizedDataset('train',
                                              res=cfg.input_res,
                                              pointcloud_samples=cfg.num_points,
                                              data_path=cfg.data_dir,
                                              split_file=cfg.split_file,
                                              batch_size=cfg.batch_size,
                                              num_sample_points=cfg.num_sample_points_training,
                                              num_workers=30,
                                              sample_distribution=cfg.sample_ratio,
                                              sample_sigmas=cfg.sample_std_dev)
'''

# -------------------------
# Shapegan Airplane Dataset
# -------------------------

root = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/airplanes'
train_dataset = PointDataset.from_split(root, split='train')


# -------------------------
# NDF Val Dataset
# -------------------------
val_dataset = voxelized_data.VoxelizedDataset('val',
                                              res=cfg.input_res,
                                              pointcloud_samples=cfg.num_points,
                                              data_path=cfg.data_dir,
                                              split_file=cfg.split_file,
                                              batch_size=cfg.batch_size,
                                              num_sample_points=cfg.num_sample_points_training,
                                              num_workers=30,
                                              sample_distribution=cfg.sample_ratio,
                                              sample_sigmas=cfg.sample_std_dev)


trainer = GAN_training_airplanes.Trainer(netG,
                               netD,
                               torch.device("cuda:1"),
                               train_dataset,
                               val_dataset,
                               cfg.exp_name,
                               optimizerG='RMSprop',
                               optimizerD='RMSprop',
                               lr=cfg.lr)

trainer.train_model(cfg.num_epochs)
