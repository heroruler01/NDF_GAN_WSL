import os
import trimesh
from tqdm import tqdm
import numpy as np
import traceback

# ================= 1. 路径配置 (请确保这里指向正确) =================
# 项目根目录
PROJECT_ROOT = '/mnt/c/Users/heror/Desktop/project'

# 输入: 你的原始 obj 模型文件夹
INPUT_DATASET_NAME = '00_good'
DIRECTORY_MODELS = os.path.join(PROJECT_ROOT, 'data/shapenet', INPUT_DATASET_NAME)

# 输出: 生成后的 Watertight (含内部结构) 模型文件夹
OUTPUT_DATASET_NAME = '00_good_watertight'
DIRECTORY_INTERNAL = os.path.join(PROJECT_ROOT, 'data/shapenet', OUTPUT_DATASET_NAME)

MODEL_EXTENSION = '.obj'

# ================= 2. 辅助函数 (内联 util 和 SDF 库) =================

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 尝试导入 mesh_to_sdf 的缩放函数，如果没有就用备选方案
try:
    from mesh_to_sdf import scale_to_unit_sphere
except ImportError:
    print("⚠️ 警告: 未找到 mesh_to_sdf，使用简易缩放逻辑。")
    def scale_to_unit_sphere(mesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()
        vertices = mesh.vertices - mesh.bounding_box.centroid
        distances = np.linalg.norm(vertices, axis=1)
        mesh.vertices /= np.max(distances)
        return mesh

def get_model_files():
    # 递归查找输入目录下的所有 .obj 文件
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_hash(filename):
    # 提取模型名称 (去除路径和后缀)
    # 例如: .../chair_001.obj -> chair_001
    return os.path.splitext(os.path.basename(filename))[0]

def get_internal_filename(model_filename):
    # 构建输出路径: output_dir/模型名/model.ply
    return os.path.join(DIRECTORY_INTERNAL, get_hash(model_filename), 'model.ply')

def get_internal_directory(model_filename):
    return os.path.join(DIRECTORY_INTERNAL, get_hash(model_filename))

# ================= 3. 核心逻辑 (制造“内脏”) =================

def downscale(mesh):
    """
    创建一个极其微小的内部网格，放在物体中心。
    这有助于 SDF 训练，告诉网络“这里是绝对的内部”。
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    # 1. 归一化重心
    vertices = mesh.vertices - mesh.bounding_box.centroid
    # 2. 计算距离
    distances = np.linalg.norm(vertices, axis=1)
    # 3. 疯狂缩小 (缩小到原来的 1/3 再除以最大距离)
    vertices /= (np.max(distances) * 3)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def process_model_file(filename):
    try:
        # 1. 检查是否已经生成过
        if os.path.exists(get_internal_filename(filename)):
            # print(f"跳过已存在文件: {get_hash(filename)}")
            return

        # 2. 加载原始模型
        mesh = trimesh.load(filename)
        
        # 3. 确保输出目录存在
        ensure_directory(get_internal_directory(filename))

        # 4. 变成单位球 (标准化)
        mesh = scale_to_unit_sphere(mesh)

        # 5. 生成缩小版内部网格 (Internal Mesh)
        mesh_s = downscale(mesh)

        # 6. 合并 (外壳 + 内核)
        combined = trimesh.util.concatenate(mesh, mesh_s)

        # 7. 导出为 PLY
        combined.export(file_obj=get_internal_filename(filename))
        
    except Exception as e:
        print(f"❌ 处理出错 {filename}: {e}")
        traceback.print_exc()

def process_model_files():
    print(f"📂 输入: {DIRECTORY_MODELS}")
    print(f"📂 输出: {DIRECTORY_INTERNAL}")
    ensure_directory(DIRECTORY_INTERNAL)
    
    files = list(get_model_files())
    
    if not files:
        print("❌ 未找到 .obj 文件，请检查输入路径！")
        return

    print(f"检测到 {len(files)} 个文件，开始处理 (单进程模式)...")

    # ❌ 移除多进程 Pool，防止 WSL 死锁
    # ✅ 改用普通 tqdm 循环
    progress = tqdm(total=len(files))
    
    for filename in files:
        process_model_file(filename)
        progress.update()
        
    print("\n✅ 处理完成！生成的 .ply 文件位于:", DIRECTORY_INTERNAL)

if __name__ == '__main__':
    process_model_files()