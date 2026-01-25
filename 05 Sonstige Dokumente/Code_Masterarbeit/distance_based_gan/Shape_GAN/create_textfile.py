import os

# train.txt wird in Shape_Gan Verzeichnis erstellt
DIRECTORY_MODELS = '/home/sebastian/ndf_gan/Shape_GAN/data/shapenet/00_good_preprocessed/uniform'
MODEL_EXTENSION = '.npy'

lines = []
def get_model_filenames():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                lines.append(filename.split('.')[-2]) #eigentlich -2

    with open('train.txt', 'w') as f:
        f.write('\n'.join(lines))




get_model_filenames()
