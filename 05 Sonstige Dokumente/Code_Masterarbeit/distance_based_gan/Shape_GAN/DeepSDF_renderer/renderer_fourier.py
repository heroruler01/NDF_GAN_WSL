import os.path as osp
import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import open3d as op
import glob
from torch.utils.data import DataLoader


from Shape_GAN.datasets import PointDataset
from Shape_GAN.point_sdf_net import PointNet, SDFGenerator
from Shape_GAN.fourier_feature import FourierFeatures
#np.random.seed(0)
#torch.manual_seed(0)
checkpoint_path = "/home/sebastian/ndf_gan/experiments/Pointnet_max_fourier_G/"
# checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/PointNet_Mix_standardcat/"
# for standard shapegan architecture with their data!:
# checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoint_rendering/"
# for standard shapegan architecture with their data but ABS:
# checkpoint_path = "/home/sebastian/ndf_gan/experiments/shapenetall/checkpoints_shapeGAN_NDF_Sphere_abs/"
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

LATENT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True
FOURIER = True
N = 256
max_batch = 1024

z = torch.randn(1, 128, device=device)
# z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
generator = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)

voxel_origin = [-1.1, -1.1, -1.1]
voxel_size = 2.2 / (256 - 1)

ply_filename = "testobject.ply"

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

if FOURIER:
    encoding = FourierFeatures(frequency_matrix)

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor, # fake
    voxel_grid_origin, # voxel_origin
    voxel_size, # voxel:size
    ply_filename_out, # ply_filename
    offset=None, # einfach in Aufruf offset
    scale=None, #  in Aufruf scale
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n) --> was sagen mir die (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.detach().numpy()

    # spacing: Voxel spacing in spatial dimensions corresponding to numpy array indexing dimensions (M, N, P) as in volume.
    # level: Contour value to search for isosurfaces in volume. If not given or None, the average of the min and max of vol is used.
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.045#, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

'''
root = osp.join(f'data') # hier kann ich ja ein paar uniform rein kopieren und dann praktisch eigenes Dataset draus machen
#dataset = PointDataset.from_split(root, split='train')
loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=6)

for uniform, _ in loader:

    uniform = uniform.to(device)
    u_pos, u_dist = uniform[..., :3], uniform[..., 3:]
'''
# root = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/airplanes'
# dataset = PointDataset.from_split(root, split='train')

# loader = DataLoader(dataset, 1, shuffle=True, num_workers=6)


overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
samples = torch.zeros(N ** 3, 4)

# transform first 3 columns --> wird hier ein grid erzeugt??
# to be the x, y, z index
samples[:, 2] = overall_index % N
samples[:, 1] = (overall_index.long() / N) % N
samples[:, 0] = ((overall_index.long() / N) / N) % N

# transform first 3 columns
# to be the x, y, z coordinate
samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]


'''
# Just for visualization of the sphere grid! 
samples = samples.detach().cpu().numpy()
samples = samples[:,:3]
pcd = op.geometry.PointCloud()
pcd.points = op.utility.Vector3dVector(samples)
op.visualization.draw_geometries([pcd])
'''

num_samples = N ** 3

samples.requires_grad = False

head = 0

while head < num_samples:
    print((head/num_samples) * 100)
    sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(device) # einfach ein subset an samples, (32)

    sample_subset = encoding(sample_subset)

    samples[head : min(head + max_batch, num_samples), 3] = (
        generator(sample_subset, z).squeeze().detach().cpu()
    )
    head += max_batch

sdf_values = samples[:, 3]
test_min = np.min(sdf_values.detach().numpy())
test_max = np.max(sdf_values.detach().numpy())
sdf_values = sdf_values.reshape(N, N, N)

convert_sdf_samples_to_ply(
    sdf_values.data.cpu(),
    voxel_origin,
    voxel_size,
    ply_filename,
    offset=True,
    scale=True,
)


'''
for uniform, _ in loader:

    uniform = uniform.to(device)
    u_pos, u_dist = uniform[..., :3], uniform[..., 3:]

    fake = generator(u_pos, z)

    convert_sdf_samples_to_ply(fake, voxel_origin, voxel_size, ply_filename, offset=True, scale=True)

'''
