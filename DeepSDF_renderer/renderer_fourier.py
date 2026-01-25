import sys
import os
import time
import torch
import numpy as np
import plyfile
import skimage.measure
import logging

# ================= 1. 路径与环境修复 =================
# 将上级目录加入路径，这样才能找到 point_sdf_net.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from point_sdf_net import SDFGenerator
from fourier_feature import FourierFeatures

# ================= 2. 配置部分 =================
# 指向模型路径
CHECKPOINT_PATH = os.path.join(parent_dir, "models", "experiments", "pointnet_mix_fourier_selected1000", "checkpoint.tar")
# 输出文件名
OUTPUT_FILENAME = os.path.join(parent_dir, "generated_car.ply")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 必须与训练参数一致
LATENT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True
N = 256 # 分辨率
max_batch = 64 ** 3 

# ================= 3. 加载模型 =================
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ 找不到模型文件: {CHECKPOINT_PATH}")

    print(f'📂 Loading checkpoint from: {CHECKPOINT_PATH}')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    generator = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
    generator.load_state_dict(checkpoint['netG_state_dict'])
    frequency_matrix = checkpoint['frequency_matrix']
    
    return generator, frequency_matrix

generator, frequency_matrix = load_checkpoint()
generator = generator.to(device)
generator.eval() 

encoding = FourierFeatures(frequency_matrix)

# ================= 4. 生成 3D 网格 (带自动纠错) =================
def convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor, voxel_grid_origin, voxel_size, ply_filename_out):
    print("🔨 Running Marching Cubes (Surface Extraction)...")
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.detach().cpu().numpy()

    # --- 🟢 关键修复：自动寻找合适的 Level ---
    min_val = numpy_3d_sdf_tensor.min()
    max_val = numpy_3d_sdf_tensor.max()
    print(f"📊 Data Stats -> Min: {min_val:.5f}, Max: {max_val:.5f}")

    # 默认寻找 0.02 处的表面
    target_level = 0.02
    
    # 如果最小值都比 0.02 大，说明车子比较“瘦”，我们需要提高标准，否则会报错
    if min_val > target_level:
        target_level = min_val + 0.005
        print(f"⚠️ Notice: Increasing surface level to {target_level:.5f} to match data range.")

    try:
        # 尝试提取表面
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(numpy_3d_sdf_tensor, level=target_level)
        except AttributeError:
            verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(numpy_3d_sdf_tensor, level=target_level)
    except ValueError as e:
        print(f"❌ Marching Cubes Failed: {e}")
        return 

    # 坐标转换
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # 保存 PLY
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
    ply_data.write(ply_filename_out)

    print(f"✅ Success! Saved mesh to {ply_filename_out}")
    print(f"⏱️ Time taken: {time.time() - start_time:.2f}s")

# ================= 5. 主程序 =================
if __name__ == "__main__":
    voxel_origin = [-1.1, -1.1, -1.1]
    voxel_size = 2.2 / (N - 1)

    print("⚡ Generating grid samples...")
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    samples.requires_grad = False
    head = 0

    # 🟢 随机种子：修改这个数字可以生成不同的车
    torch.manual_seed(42) 
    z = torch.randn(1, LATENT_SIZE, device=device)

    print("🧠 Querying Generator (Inference)...")
    with torch.no_grad():
        while head < num_samples:
            if head % (max_batch * 10) == 0:
                print(f"Progress: {(head/num_samples)*100:.1f}%")
                
            batch_end = min(head + max_batch, num_samples)
            sample_subset = samples[head : batch_end, 0:3].to(device)

            # 🟢 [注意] 这里不能有 encoding()，因为生成器直接吃坐标
            pred = generator(sample_subset, z).squeeze()
            samples[head : batch_end, 3] = pred.cpu()
            
            head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    convert_sdf_samples_to_ply(
        sdf_values,
        voxel_origin,
        voxel_size,
        OUTPUT_FILENAME
    )