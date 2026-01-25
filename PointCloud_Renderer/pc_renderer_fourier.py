import sys
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import open3d as op

# ================= 1. 路径修复 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from point_sdf_net import SDFGenerator
from fourier_feature import FourierFeatures 

# ================= 2. 配置部分 =================
CHECKPOINT_PATH = os.path.join(parent_dir, "models", "experiments", "pointnet_mix_fourier_selected1000", "checkpoint.tar")
OUTPUT_DIR = os.path.join(parent_dir, "generated_pointclouds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

LATENT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

# ================= 3. 加载模型 =================
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ 找不到模型: {CHECKPOINT_PATH}")
        
    print(f'📂 Loading checkpoint from: {CHECKPOINT_PATH}')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    generator = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
    generator.load_state_dict(checkpoint['netG_state_dict'])
    
    return generator

generator = load_checkpoint()
generator = generator.to(device)
generator.eval() 

# ================= 4. 核心生成逻辑 =================
# 🟢 [修改] 默认 filter_val 改为 0.05 (放宽10倍)，num_steps 增加到 30
# 将 filter_val 改得极小，迫使它只保留真正的核心
def generate_point_cloud(seed=42, num_steps=30, target_num_points=30000, filter_val=0.00005):
    print(f"⚡ Start generating (Seed: {seed}, Threshold: {filter_val})...")
    start_time = time.time()
    
    torch.manual_seed(seed) 
    z = torch.randn(1, LATENT_SIZE, device=device)

    final_points = np.zeros((0, 3))
    batch_size = 100000 
    
    iteration = 0
    while len(final_points) < target_num_points:
        # A. 随机撒点
        points = torch.randn(1, batch_size, 3, device=device)
        points = points / torch.norm(points, dim=-1, keepdim=True) 
        points = points * (torch.rand(1, batch_size, 1, device=device) ** (1/3)) 
        
        points.requires_grad = True

        # B. 梯度下降 (推向表面)
        for j in range(num_steps):
            sdf_pred = generator(points, z)
            sdf_pred.sum().backward()
            grad = points.grad
            
            with torch.no_grad():
                grad_norm = F.normalize(grad, dim=-1)
                points = points - grad_norm * sdf_pred
            
            points = points.detach()
            points.requires_grad = True

        # C. 筛选合格的点
        with torch.no_grad():
            final_sdf = generator(points, z)
            
            # 🟢 [新增] 调试打印：让我们看看距离到底是多少？
            if iteration % 5 == 0:
                min_dist = torch.abs(final_sdf).min().item()
                mean_dist = torch.abs(final_sdf).mean().item()
                print(f"   [Debug] Iter {iteration}: Min Dist = {min_dist:.5f}, Mean Dist = {mean_dist:.5f}")

            # 筛选
            mask = torch.abs(final_sdf) < filter_val
            mask = mask.squeeze()
            
            valid_points = points.squeeze()[mask].cpu().numpy()
            
            if len(valid_points) > 0:
                final_points = np.vstack((final_points, valid_points))
                
        print(f"  Iteration {iteration}: Collected {len(final_points)} / {target_num_points} points")
        iteration += 1
        
        # 🟢 [修改] 增加最大迭代次数防止死循环
        if iteration > 100:
            print("⚠️ Reached max iterations. Changing logic slightly...")
            # 如果实在找不到点，就稍微再放宽一点点标准继续跑
            filter_val += 0.01
            print(f"⚠️ Increased threshold to {filter_val}")

    duration = time.time() - start_time
    return final_points, duration

# ================= 5. 批量生成 =================
if __name__ == "__main__":
    
    NUM_CARS = 5
    print(f"🚀 Starting batch generation of {NUM_CARS} cars...")

    for i in range(NUM_CARS):
        current_seed = int(time.time() * 1000) % 10000 + i * 100
        
        # 🟢 [修改] 这里传入更宽松的阈值
        points, duration = generate_point_cloud(seed=current_seed, filter_val=0.06)

        filename = f"car_{i}_seed{current_seed}.ply"
        save_path = os.path.join(OUTPUT_DIR, filename)

        pcd = op.geometry.PointCloud()
        pcd.points = op.utility.Vector3dVector(points)
        op.io.write_point_cloud(save_path, pcd)

        print(f"✅ [{i+1}/{NUM_CARS}] Saved: {filename} (Time: {duration:.2f}s)\n")
    
    print(f"🎉 All done! Check folder: {OUTPUT_DIR}")