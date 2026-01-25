import os
import trimesh
from tqdm import tqdm
import numpy as np
import open3d as op
import shutil
from util import ensure_directory
from multiprocessing import Pool
import traceback
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, scale_to_unit_sphere, BadMeshException
import igl


DATASET_NAME = 'cars_internal_final'
# DIRECTORY_MODELS = '/home/sebastian/ndf-master/datasets/shapenet/data/02958343' # directory of all models
DIRECTORY_MODELS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/cars_internal_tight'
MODEL_EXTENSION = '.obj'
DIRECTORY_INTERNALS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/{:s}/'.format(DATASET_NAME)


def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)


def get_hash(filename): # gives back folder of model.py
    return filename.split('/')[-2]


def get_distance_values(filename):
    mesh = op.io.read_triangle_mesh(filename)
    convexhull = mesh.compute_convex_hull()[0]
    convexPC = convexhull.sample_points_uniformly(number_of_points=10000, seed=- 1)
    normalPC = mesh.sample_points_uniformly(number_of_points=10000, seed=-1)
    normalPC.paint_uniform_color([160 / 255, 160 / 255, 160 / 255])

    # op.visualization.draw_geometries([normalPC])

    distance = normalPC.compute_point_cloud_distance(convexPC)
    distance = np.mean(distance)

    return distance

def compare_distance_values(filename):
    threshold = 0.029
    distance = get_distance_values(filename)
    if distance > threshold:
        target_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename) + '/model.obj')

        temp_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename))
        if os.path.exists(temp_path) == False:
            os.mkdir(temp_path)

        shutil.copyfile(filename, target_path)

def compare_nearest_neighbors(filename):
    sample_num = 50000
    threshold = 115
    counter = []
    mesh = op.io.read_triangle_mesh(filename)
    normalPC = mesh.sample_points_uniformly(number_of_points=sample_num, seed=-1)
    pcd_tree = op.geometry.KDTreeFlann(normalPC)
    for i in range(50000):
        [k, idx, distances] = pcd_tree.search_radius_vector_3d(normalPC.points[i], 0.025)
        counter.append(int(k))
    counter_mean = np.mean(counter)

    if counter_mean < threshold:
        target_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename) + '/model.obj')

        temp_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename))
        if os.path.exists(temp_path) == False:
            os.mkdir(temp_path)

        shutil.copyfile(filename, target_path)


if __name__ == '__main__':

    files = list(get_model_files())

    # Parallelize on CPU
    worker_count = os.cpu_count() // 2
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(worker_count)

    progress = tqdm(total=len(files))

    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(compare_distance_values, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()



