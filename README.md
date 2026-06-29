
🛠️ Data Processing Pipeline for NDF-GAN

This module is responsible for converting raw 3D meshes (ShapeNet `.obj` files) into Neural Distance Fields (NDF) point clouds (`.npy` format) required for training the GAN.
📋 Environment & Configuration

1. Prerequisites
* **OS**: Linux or Windows Subsystem for Linux (WSL2).
* **Python**: 3.9 (Recommended).
* **Dependencies**:
    ```bash
    conda create -n ndf_gan python=3.9
    conda activate ndf_gan
    pip install numpy torch trimesh open3d mesh_to_sdf scipy tqdm
    ```

 2. ⚠️ Critical Path Configuration
Before running any scripts, you **MUST** update the `PROJECT_ROOT` variable in all Python scripts (`prepare_shapenet_dataset.py`, `create_textfile.py`, etc.) to match your local environment.


In prepare_shapenet_dataset.py
PROJECT_ROOT = '/mnt/c/Users/your_username/Desktop/project'
Ensure paths point to: PROJECT_ROOT + /data/shapenet/...

🚀 Usage Guide
The pipeline consists of three main steps. Run them in the following order:

# or just run Workflow Automation:run_pipeline.sh


Step 1: Preprocessing & SDF Generation
Converts .obj meshes into Signed Distance Field (SDF) samples. This script generates uniform points (for volume) and surface points (for details).

Script: prepare_shapenet_dataset.py

Input: data/shapenet/00_good (Raw .obj files)

Output: data/shapenet/00_good_preprocessed

Usage:

Bash
Note: In WSL, if the script freezes, ensure multiprocessing is disabled or num_workers=0 inside the script.
python prepare_shapenet_dataset.py

Insertion for WatertightnessInserts an internal kernel into the meshes to create "watertight" volumes, ensuring the network can learn the object's interior.Script: create_watertight_meshes.py (corresponds to prepare_watertight_dataset)Output: data/shapenet/00_good_watertight
 

Automated Model SelectionFilters raw ShapeNet models to keep only non-watertight models with sufficient geometric complexity (internal details).Script: prepare_internal_detail.py < Input: data/shapenet/00_good (Raw)  Output: data/shapenet/00_good (Selected)1.2 

 Generate Dataset Index
Scans the preprocessed folder and creates a train.txt manifest file required by the training loader.

Script: create_textfile.py

Output: data/shapenet/00_good_preprocessed/train.txt

Usage:

Bash
python create_textfile.py
📊 Expected Results & Directory Structure
After running the pipeline, your data directory should be structured as follows:

Plaintext
data/shapenet/00_good_preprocessed/
├── uniform/            # Global volumetric sampling points
│   ├── model_01.npy    
│   └── ...
├── surface/            # Near-surface high-detail sampling points
│   ├── model_01.npy    
│   └── ...
└── train.txt           # Index file containing model IDs (e.g., model_01)
Data Format (.npy)
Each .npy file contains a NumPy array with shape (N, 4):

Columns 0-2: (x, y, z) coordinates (Normalized to unit sphere).

Column 3: Distance value (SDF).

< 0: Inside the object.

> 0: Outside the object.
<img width="805" height="785" alt="image" src="https://github.com/user-attachments/assets/5ef2b7ab-b9dd-4100-9851-9503e2020229" />
<img width="1499" height="887" alt="image" src="https://github.com/user-attachments/assets/301b8be8-0ad0-48d3-882c-65df728b088d" />
