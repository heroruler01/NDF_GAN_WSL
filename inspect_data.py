import numpy as np
import os
import sys

# ✅ [关键修复 1] 强制使用无窗口模式 (必须放在 import pyplot 之前)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("🚀 脚本开始运行...")

# ================= 1. 基础配置 =================
PROJECT_ROOT = '/mnt/c/Users/heror/Desktop/project'
DATASET_NAME = '00_good_preprocessed'
TARGET_DIR = os.path.join(PROJECT_ROOT, 'data/shapenet', DATASET_NAME, 'surface')

# ================= 2. 核心逻辑 =================
def inspect_latest_file():
    print(f"📂 正在检查目录: {TARGET_DIR}")
    
    if not os.path.exists(TARGET_DIR):
        print(f"❌ 错误：找不到目录 {TARGET_DIR}")
        return

    files = [f for f in os.listdir(TARGET_DIR) if f.endswith('.npy')]
    if len(files) == 0:
        print("❌ 目录里是空的，没有 .npy 文件！")
        return
    
    print(f"✅ 发现 {len(files)} 个数据文件，正在寻找最新的...")
    
    # 按时间排序
    files.sort(key=lambda x: os.path.getmtime(os.path.join(TARGET_DIR, x)))
    filename = files[-1]
    filepath = os.path.join(TARGET_DIR, filename)
    
    print("-" * 40)
    print(f"🔎 锁定最新文件: {filename}")
    print("-" * 40)

    # 加载数据
    try:
        data = np.load(filepath)
        print("✅ 数据加载成功！")
    except Exception as e:
        print(f"❌ 无法加载文件: {e}")
        return

    points = data[:, :3]
    sdf_values = data[:, 3]

    print(f"📊 数据点总数: {points.shape[0]}")
    
    # 调用绘图
    visualize_3d(points, sdf_values, filename)

# ================= 3. 绘图逻辑 =================
def visualize_3d(points, sdf_values, filename_title):
    print("🎨 正在初始化 3D 画布...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ✅ [关键修复 2] 稍微降低一点点数，5万点在纯CPU下可能要跑1-2分钟
    # 先用 10,000 个点测试，确保能跑通。如果很快，你可以自己改成 50,000
    DISPLAY_LIMIT = 10000 

    if len(points) > DISPLAY_LIMIT:
        print(f"⚠️ 为了速度，随机抽取 {DISPLAY_LIMIT} 个点进行绘制...")
        indices = np.random.choice(len(points), DISPLAY_LIMIT, replace=False)
        points = points[indices]
        sdf_values = sdf_values[indices]

    print("🖌️ 正在绘制散点 (Matplotlib Scatter)... 这步最慢，请耐心等待 10-20 秒...")
    img = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                     c=sdf_values, cmap='coolwarm', s=1.0, alpha=0.5)
    
    ax.set_title(f'Preview: {filename_title}')
    plt.colorbar(img, label='SDF Value')
    
    save_path = 'inspect_result_HD.png'
    print(f"💾 正在保存图片到: {save_path}")
    
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    print("-" * 40)
    print(f"✅ 成功！图片已保存。请在左侧文件列表打开 {save_path}")

if __name__ == "__main__":
    inspect_latest_file()