import os
import trimesh
from tqdm import tqdm
import numpy as np
from util import ensure_directory
from multiprocessing import Pool, cpu_count
import traceback
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, scale_to_unit_sphere, BadMeshException
import igl
import open3d as op



DATASET_NAME = 'cars_watertight_internal'
DIRECTORY_MODELS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/02958343'#/home/sebastian/ndf-master/datasets/shapenet/data/02958343
MODEL_EXTENSION = '.obj'
DIRECTORY_INTERNAL = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/{:s}/'.format(DATASET_NAME)


def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)


def get_hash(filename):
    return filename.split('/')[-2]


def get_internal_filename(model_filename):
    return os.path.join(DIRECTORY_INTERNAL, get_hash(model_filename) + '/model.ply')


def get_internal_directory(model_filename):
    return os.path.join(DIRECTORY_INTERNAL, get_hash(model_filename))


def downscale(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= (np.max(distances) * 3)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def process_model_file(filename):
    mesh = trimesh.load(filename)
    if not os.path.exists(get_internal_filename(filename)):
        os.makedirs(get_internal_directory(filename))
        mesh = scale_to_unit_sphere(mesh)
        mesh_s = downscale(mesh)
        test = get_internal_filename(filename)
        combined = trimesh.util.concatenate(mesh, mesh_s)

        # filnema existiert nicht,d as directory muss erst erstellt werden!!
        combined.export(file_obj=get_internal_filename(filename))



def process_model_files():
    ensure_directory(DIRECTORY_INTERNAL)
    files = list(get_model_files())

    worker_count = os.cpu_count() // 2
    print("Using {:d} processes.".format(worker_count))
    pool = Pool(max(cpu_count()//2, 1))

    progress = tqdm(total=len(files))

    def on_complete(*_):
        progress.update()

    for filename in files:
        pool.apply_async(process_model_file, args=(filename,), callback=on_complete)
    pool.close()
    pool.join()


if __name__ == '__main__':
    process_model_files()


