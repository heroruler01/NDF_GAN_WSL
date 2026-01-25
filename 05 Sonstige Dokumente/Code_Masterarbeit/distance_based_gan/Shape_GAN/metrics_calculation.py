import os
import open3d as op
import shutil
import numpy as np
import trimesh
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, scale_to_unit_sphere, BadMeshException

DATASET_NAME = 'cars_manually_selected'  # folder with this name must be created beforehand
DIRECTORY_MODELS = '/home/sebastian/ndf_gan/Shape_GAN/test_generation'
MODEL_EXTENSION = '.ply'
DIRECTORY_INTERNALS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/{:s}/'.format(DATASET_NAME)  # path to new dataset

 # GT
DIRECTORY_MODELS_GT = '/home/sebastian/ndf_gan/Shape_GAN/test_set'
MODEL_EXTENSION_GT = '.ply'


def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in sorted(files):
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_model_files_GT():
    for directory, _, files in os.walk(DIRECTORY_MODELS_GT):
        for filename in sorted(files):
            if filename.endswith(MODEL_EXTENSION_GT):
                yield os.path.join(directory, filename)


def get_hash(filename):
    return filename.split('/')[-2]


def calculate_chamfer_distance(filename, filename_gt):
    # mesh = op.io.read_triangle_mesh(filename)
    # mesh_gt = op.io.read_triangle_mesh(filename_gt)
    mesh_gt = trimesh.load(filename_gt)
    mesh = trimesh.load(filename)
    mesh = scale_to_unit_sphere(mesh)
    mesh_gt = scale_to_unit_sphere(mesh_gt)
    mesh_gt = mesh_gt.as_open3d
    mesh = mesh.as_open3d
    pc = mesh.sample_points_uniformly(number_of_points=10000, seed=-1)
    pc_gt = mesh_gt.sample_points_uniformly(number_of_points=10000, seed=-1)

    distance = pc.compute_point_cloud_distance(pc_gt)
    distance = np.mean(distance)

    distance_list.append(distance)

    return distance_list


def evaluate_model(filename):

    target_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename) + '/model.obj')
    temp_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename))
    if os.path.exists(temp_path) == False:
        os.mkdir(temp_path)

        shutil.copyfile(filename, target_path)


if __name__ == '__main__':


    files = list(get_model_files())
    files_gt = list(get_model_files_GT())

    index_list = []
    for filename in files:
        distance_list = []
        print(filename)
        for filename_gt in files_gt:
            calculate_chamfer_distance(filename, filename_gt)
        index_max = min(range(len(distance_list)), key=distance_list.__getitem__)
        index_list.append(index_max)
    index_set = set(index_list)
    unique_values = len(index_set)
    percentage = unique_values/len(files_gt)
    print(percentage)
    with open('readme.txt', 'w') as f:
        f.write(str(percentage))
    print("finished")
