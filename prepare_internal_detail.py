import os
import shutil
import numpy as np
from tqdm import tqdm
import traceback

# 尝试导入 Open3D
try:
    import open3d as op
    # 让 Open3D 保持安静
    op.utility.set_verbosity_level(op.utility.VerbosityLevel.Error)
except ImportError:
    print("❌ 严重错误: 未安装 Open3D。请运行: pip install open3d")
    exit()

# ================= 1. 路径配置 =================
# 项目根目录
PROJECT_ROOT = '/mnt/c/Users/heror/Desktop/project'

# 输入: 你的原始模型文件夹
INPUT_DATASET_NAME = '00_good'
DIRECTORY_MODELS = os.path.join(PROJECT_ROOT, 'data/shapenet', INPUT_DATASET_NAME)

# 输出: 筛选后的精华文件夹
OUTPUT_DATASET_NAME = '00_good_selected'
DIRECTORY_INTERNALS = os.path.join(PROJECT_ROOT, 'data/shapenet', OUTPUT_DATASET_NAME)

MODEL_EXTENSION = '.obj'

# ================= 2. 辅助函数 =================

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_hash(filename):
    return os.path.splitext(os.path.basename(filename))[0]

# ================= 3. 核心筛选逻辑 (修复了 Open3D 参数) =================

def get_distance_values(filename):
    """
    计算 [原始网格] 与 [它的凸包] 之间的平均距离。
    """
    # 读取网格
    mesh = op.io.read_triangle_mesh(filename)
    
    if not mesh.has_vertices():
        return 0.0

    # 1. 计算凸包
    try:
        convexhull = mesh.compute_convex_hull()[0]
    except Exception:
        return 0.0
    
    # 2. 采样点云 (✅ 修复：移除了不支持的 seed 参数)
    # 降低采样数以提高速度，同时避免参数错误
    convexPC = convexhull.sample_points_uniformly(number_of_points=5000)
    normalPC = mesh.sample_points_uniformly(number_of_points=5000)
    
    # 3. 计算距离
    distance = normalPC.compute_point_cloud_distance(convexPC)
    distance = np.mean(distance)

    return distance

def process_file(filename):
    # 阈值：0.029 是经验值。
    THRESHOLD = 0.029 
    
    try:
        distance = get_distance_values(filename)
        
        # 如果分数够高 (细节够丰富)
        if distance > THRESHOLD:
            model_name = get_hash(filename)
            target_dir = os.path.join(DIRECTORY_INTERNALS, model_name)
            ensure_directory(target_dir)
            
            target_path = os.path.join(target_dir, 'model.obj')
            shutil.copyfile(filename, target_path)
            
    except Exception as e:
        # 只打印简短错误
        tqdm.write(f"⚠️ 跳过: {os.path.basename(filename)} - {str(e)[:50]}...")

# ================= 4. 主程序 =================

if __name__ == '__main__':
    print("-" * 50)
    print(f"📂 输入目录: {DIRECTORY_MODELS}")
    print(f"📂 输出目录: {DIRECTORY_INTERNALS}")
    print("-" * 50)
    
    ensure_directory(DIRECTORY_INTERNALS)

    files = list(get_model_files())
    
    if not files:
        print("❌ 错误：在输入目录没找到 .obj 文件！")
        exit()

    print(f"检测到 {len(files)} 个模型，开始筛选...")

    progress = tqdm(total=len(files))

    for filename in files:
        process_file(filename)
        progress.update()
    
    # 统计结果
    selected_count = 0
    for _, dirs, _ in os.walk(DIRECTORY_INTERNALS):
        selected_count += len(dirs)
        
    print("\n" + "=" * 50)
    print(f"✅ 筛选完成！")
    print(f"📥 原始数量: {len(files)}")
    print(f"📤 保留数量: {selected_count}")
    print(f"💾 结果已保存在: {DIRECTORY_INTERNALS}")
    print("=" * 50)