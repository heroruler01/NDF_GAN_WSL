import os
import random
import trimesh
import shutil  # ✅ 新增：用于删除文件夹

# ================= 配置路径 =================
SOURCE_DIR = "/mnt/c/Users/heror/Desktop/project/part_retrieval/Part Retrieval CIRP Annals/STL"
DEST_DIR = "/mnt/c/Users/heror/Desktop/project/data/shapenet/00_good"

def clean_directory(directory):
    """暴力清空一个文件夹"""
    if os.path.exists(directory):
        print(f"🧹 正在清空旧数据: {directory}")
        # 删除整个文件夹树
        shutil.rmtree(directory)
    # 重新创建空文件夹
    os.makedirs(directory)

def convert_one_random_file():
    # 1. ✅ 先清空输出目录！(核心修改)
    clean_directory(DEST_DIR)

    # 2. 获取源文件列表
    try:
        all_files = os.listdir(SOURCE_DIR)
        stl_files = [f for f in all_files if f.lower().endswith('.stl')]
    except FileNotFoundError:
        print("❌ 找不到源目录")
        return

    if not stl_files:
        print("❌ 没有 STL 文件")
        return

    # 3. 随机选择
    selected_file = random.choice(stl_files)
    src_path = os.path.join(SOURCE_DIR, selected_file)
    
    # 构建输出路径
    file_name_no_ext = os.path.splitext(selected_file)[0]
    dst_filename = file_name_no_ext + ".obj"
    dst_path = os.path.join(DEST_DIR, dst_filename)

    print("-" * 40)
    print(f"🎲 选中文件: {selected_file}")
    
    # 4. 转换
    try:
        mesh = trimesh.load(src_path)
        mesh.export(dst_path, file_type='obj')
        print(f"✅ 转换完成。文件夹已重置，当前仅包含此文件。")
    except Exception as e:
        print(f"❌ 失败: {e}")

if __name__ == "__main__":
    convert_one_random_file()