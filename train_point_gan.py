import os.path as osp
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import RMSprop
import random
from datasets import PointDataset
from point_sdf_net import PointNet, SDFGenerator

from torch.utils.tensorboard import SummaryWriter
# import open3d as op # ❌ [WSL优化] 注释掉 Open3D，训练时不需要绘图，防止报错

from fourier_feature import FourierFeatures

# ================= 1. 环境与路径配置 (关键修改) =================
# 🟢 [修改] 强制指定项目根目录
PROJECT_ROOT = '/mnt/c/Users/heror/Desktop/project'

parser = argparse.ArgumentParser()
# 运行示例: python train_point_gan.py --category shapenet/00_good_selected
parser.add_argument('--category', type=str, required=True) 
args = parser.parse_args()

# 🟢 [修改] Writer 路径指向项目 logs 文件夹
log_dir = os.path.join(PROJECT_ROOT, "logs", "runs", "pointnet_mix_fourier_selected1000")
writer = SummaryWriter(log_dir=log_dir)

# 🟢 [修改] Checkpoint 路径指向项目 models 文件夹
checkpoint_path = os.path.join(PROJECT_ROOT, "models", "experiments", "pointnet_mix_fourier_selected1000")
if not os.path.exists(checkpoint_path):
    print(f"创建存档目录: {checkpoint_path}")
    os.makedirs(checkpoint_path)

config_counter = 0

# ================= 2. 超参数配置 =================
LATENT_SIZE = 128 
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256 
NUM_LAYERS = 8 
NORM = True

# 🟢 [修改] 自动检测设备，或者默认为 cuda:0
# 原代码写死 cuda:1，这在单显卡电脑上会直接报错
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training on device: {device}")

G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Generator params: {count_parameters(G)}")
print(f"Discriminator params: {count_parameters(D)}")

G, D = G.to(device), D.to(device)
G_optimizer = RMSprop(G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

# 🟢 [修改] 数据集路径指向正确的位置
root = osp.join(PROJECT_ROOT, 'data', args.category)
# 确保数据集存在
if not os.path.exists(root):
    raise FileNotFoundError(f"❌ 数据集路径不存在: {root}")

print(f"📂 Loading dataset from: {root}")
dataset = PointDataset.from_split(root, split='train')

# 渐进式训练配置 (点数, Batch Size, Epochs)
configuration = [  
    (1024, 32, 300),
    (2048, 32, 300),
    (4096, 32, 300),
    (8192, 24, 300),
    (16384, 12, 300),
    (32768, 12, 900),
]

config_index = 0
num_steps = 0

# ---------------------------------------
# Fourier Features 配置
# ---------------------------------------
num_frequencies = 128
std_dev = 1
input_dim = 3 

if num_frequencies:
    frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                    std=std_dev).to(device)
    encoding = FourierFeatures(frequency_matrix)
else:
    encoding = torch.nn.Identity()

# ---------------------------------------
# 3. 训练主循环
# ---------------------------------------
for num_points, batch_size, epochs in configuration:
    print(f"\n--- Starting Phase: {num_points} Points | Batch Size {batch_size} | {epochs} Epochs ---")
    dataset.num_points = num_points
    
    # 🟢 [修改] WSL 稳定性修复: num_workers=0
    # 在 WSL 中，num_workers > 0 经常导致 Shared Memory 错误或死锁。
    # 设置为 0 表示只用主进程加载数据，虽然慢一点点，但绝对稳定。
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_loss_G = 0
        
        # 这里的 uniform 就是从 .npy 读取的数据
        for i, uniform in enumerate(loader):
            num_steps += 1

            uniform = uniform.to(device)
            # 切分数据：前3列是坐标(xyz)，第4列是距离值
            u_pos, u_dist = uniform[..., :3], uniform[..., 3:] 

            # 将 SDF (有正负) 转换为 NDF (只有正值)
            u_dist = torch.abs(u_dist)

            # --- Fourier Feature 变换 ---
            coordinates = u_pos
            f_features = encoding(coordinates)
            # ---------------------------

            # === 训练 Discriminator (判别器) ===
            D_optimizer.zero_grad()

            z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
            fake = G(u_pos, z) # 生成器生成假距离
            
            out_real = D(f_features, u_dist) # 判别器看真数据
            out_fake = D(f_features, fake)   # 判别器看假数据
            D_loss = out_fake.mean() - out_real.mean()

            # Gradient Penalty (WGAN-GP 核心)
            alpha = torch.rand((uniform.size(0), 1, 1), device=device)
            interpolated = alpha * u_dist + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            out = D(f_features, interpolated)

            grad = torch.autograd.grad(out, interpolated,
                                       grad_outputs=torch.ones_like(out),
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(dim=-1, p=2)
            gp = GRADIENT_PENALITY * ((grad_norm - 1).pow(2).mean())

            loss = D_loss + gp
            loss.backward()
            D_optimizer.step()

            # === 训练 Generator (生成器) ===
            # 每 5 次 D 更新 1 次 G
            if num_steps % 5 == 0:
                G_optimizer.zero_grad()
                z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
                fake = G(u_pos, z)
                out_fake = D(f_features, fake)
                loss = -out_fake.mean() # G 希望 D 给出的分数越高越好
                loss.backward()
                G_optimizer.step()

            total_loss += D_loss.abs().item()
            total_loss_G += loss.item() if isinstance(loss, torch.Tensor) else 0

        # 每个 Epoch 打印一次 Log
        print('Num points: {}, Epoch: {:03d}, D_Loss: {:.6f}'.format(
            num_points, epoch, total_loss / len(loader)))

        config_index += 1 

        writer.add_scalar('training_loss_Discriminator', total_loss / len(loader), config_index)
        writer.add_scalar('training_loss_batch_avg_Generator', total_loss_G / len(loader), config_index)

        # 保存模型存档
        path = os.path.join(checkpoint_path, 'checkpoint.tar')
        torch.save({
            'epoch': epoch, 'config_counter': config_counter, 'frequency_matrix': frequency_matrix,
            'netG_state_dict': G.state_dict(),
            'optimizerG_state_dict': G_optimizer.state_dict(),
            'netD_state_dict': D.state_dict(),
            'optimizerD_state_dict': D_optimizer.state_dict()}, path)

    config_counter += 1 

print('✅ Finished Training successfully')