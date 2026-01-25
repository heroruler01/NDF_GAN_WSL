import trimesh
import numpy as np
import open3d as op
import os

# ================= 配置 =================
# 1. 这里填你刚才生成成功的那个 PLY 文件的路径 (DeepSDF渲染出来的那个)
MESH_PATH = "generated_car.ply" 

# 2. 输出路径
OUTPUT_PATH = "generated_pointclouds/final_perfect_cloud.ply"

# 3. 你想要多少个点？
NUM_POINTS = 30000

# ================= 核心逻辑 =================
def convert_mesh_to_pcd():
    # 检查文件是否存在
    if not os.path.exists(MESH_PATH):
        print(f"❌ 错误：找不到文件 {MESH_PATH}")
        print("请确保你已经运行了 DeepSDF_renderer/renderer_fourier.py 并生成了 generated_car.ply")
        return

    print(f"📂 正在加载 Mesh 文件: {MESH_PATH} ...")
    
    # 使用 trimesh 加载网格
    mesh = trimesh.load(MESH_PATH)
    
    # 检查是否加载成功
    if mesh.is_empty:
        print("⚠️ Mesh 是空的！")
        return

    print(f"✨ Mesh 加载成功！包含 {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面")
    print(f"🔨 正在从表面采样 {NUM_POINTS} 个点...")

    # 核心步骤：直接在网格表面均匀采样
    # 这比梯度下降准确一万倍，因为它直接基于几何形状
    points, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)

    print(f"✅ 采样完成。正在保存到 {OUTPUT_PATH} ...")

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 使用 Open3D 保存为标准 .ply 点云
    pcd = op.geometry.PointCloud()
    pcd.points = op.utility.Vector3dVector(points)
    op.io.write_point_cloud(OUTPUT_PATH, pcd)

    print(f"🎉 大功告成！完美匹配形状的点云已保存。")

if __name__ == "__main__":
    convert_mesh_to_pcd()