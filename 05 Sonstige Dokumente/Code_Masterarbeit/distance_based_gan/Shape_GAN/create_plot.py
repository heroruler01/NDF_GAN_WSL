import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from tqdm import tqdm
import random
from point_sdf_net import PointNet, SDFGenerator
from glob import glob

standard_normal_distribution = torch.distributions.normal.Normal(0, 1)

checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_shapeGAN/"
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)

def load_checkpoint():
    checkpoints = glob(checkpoint_path + '/*')
    if len(checkpoints) == 0:
        print('No checkpoints found at {}'.format(checkpoint_path))
        return 0, 0

    path = checkpoint_path + 'checkpoint.tar'

    print('Loaded checkpoint from: {}'.format(path))
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['netG_state_dict'])
    return G


class ImageGrid():
    def __init__(self, width, height=1, cell_width = 3, cell_height = None, margin=0.2, create_viewer=True, crop=True):
        print("Plotting...")
        self.width = width
        self.height = height
        cell_height = cell_height if cell_height is not None else cell_width

        self.figure, self.axes = plt.subplots(height, width,
            figsize=(width * cell_width, height * cell_height),
            gridspec_kw={'left': 0, 'right': 1, 'top': 1, 'bottom': 0, 'wspace': margin, 'hspace': margin})
        self.figure.patch.set_visible(False)

        self.crop = crop
        if create_viewer:
            from rendering import MeshRenderer
            self.viewer = MeshRenderer(start_thread=False)
        else:
            self.viewer = None

    def set_image(self, image, x = 0, y = 0):
        cell = self.axes[y, x] if self.height > 1 and self.width > 1 else self.axes[x + y]
        cell.imshow(image)
        cell.axis('off')
        cell.patch.set_visible(False)

    def set_voxels(self, voxels, x = 0, y = 0, color=None):
        if color is not None:
            self.viewer.model_color = color
        self.viewer.set_voxels(voxels)
        image = self.viewer.get_image(crop=self.crop)
        self.set_image(image, x, y)

    def save(self, filename):
        plt.axis('off')
        extent = self.figure.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        plt.savefig(filename, bbox_inches=extent, dpi=400)
        if self.viewer is not None:
            self.viewer.delete_buffers()


def generate(sample_size):
    shape = torch.Size((sample_size, LATENT_SIZE))
    x = standard_normal_distribution.sample(shape).to(device)
    return x


if "gan_examples" in sys.argv:
    generator = load_checkpoint()

    COUNT = 5
    with torch.no_grad():
        voxels = generate(sample_size=COUNT)

    plot = ImageGrid(COUNT)

    for i in range(COUNT):
        plot.set_voxels(voxels[i, :, :, :], i) # [i, :]

    filename = "plots/wgan-examples.pdf" if 'wgan' in sys.argv else "plots/gan-examples.pdf"
    plot.save(filename)

if "gan_interpolation" in sys.argv:
    from util import standard_normal_distribution

    STEPS = 6

    generator = load_generator(is_wgan='wgan' in sys.argv)

    print("Generating codes...")
    with torch.no_grad():
        codes = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
        codes_start_end = standard_normal_distribution.sample((2, LATENT_CODE_SIZE))
        code_start = codes_start_end[0, :]
        code_end = codes_start_end[1, :]
        for i in range(STEPS):
            codes[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)
        voxels = generator(codes)

    plot = ImageGrid(STEPS)
    for i in range(STEPS):
        plot.set_voxels(voxels[i, :, :, :], i)

    filename = "plots/wgan-interpolation.pdf" if 'wgan' in sys.argv else "plots/gan-interpolation.pdf"
    plot.save(filename)