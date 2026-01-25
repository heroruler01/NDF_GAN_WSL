import os
import open3d as op
import shutil

# ToDo: skip models that have already been evaluated

DATASET_NAME = 'cars_manually_selected'  # folder with this name must be created beforehand
DIRECTORY_MODELS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/02958343_raw'
MODEL_EXTENSION = '.obj'
DIRECTORY_INTERNALS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/{:s}/'.format(DATASET_NAME)  # path to new dataset


def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)


def get_hash(filename):
    return filename.split('/')[-2]


def view_model(filename):
    mesh = op.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    op.visualization.draw_geometries([mesh])

    evaluate_model(filename)

def evaluate_model(filename):
    value = input("Is the model valid?\n")

    if value == "yes":
        target_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename) + '/model.obj')
        temp_path = os.path.join(DIRECTORY_INTERNALS, get_hash(filename))
        if os.path.exists(temp_path) == False:
            os.mkdir(temp_path)

        shutil.copyfile(filename, target_path)


if __name__ == '__main__':

    files = list(get_model_files())

    for filename in files:
        view_model(filename)




