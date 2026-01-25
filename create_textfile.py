import os

# ================= 1. 路径修改 (关键修改) =================
# 🔴 [旧逻辑] 依赖脚本所在位置，容易出错
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🟢 [新逻辑] 强制指定你的项目根目录 (WSL 环境)
PROJECT_ROOT = '/mnt/c/Users/heror/Desktop/project'

# 数据集名称 (必须与你运行 prepare_shapenet_dataset.py 时生成的文件夹一致)
# 如果你之前的输出叫 '00_good_preprocessed'，这里就保持不变
DATASET_NAME = '00_good_preprocessed'

# 目标路径: 训练脚本通常读取 'uniform' 或 'voxels' 文件夹里的文件名作为索引
DIRECTORY_MODELS = os.path.join(PROJECT_ROOT, 'data', 'shapenet', DATASET_NAME, 'uniform')
MODEL_EXTENSION = '.npy'

# 输出文件路径
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'train.txt')

# ================= 2. 核心逻辑 =================
def create_textfile():
    print(f"📂 正在扫描目录: {DIRECTORY_MODELS}")
    
    # 安全检查：防止目录不存在导致报错
    if not os.path.exists(DIRECTORY_MODELS):
        print(f"❌ 错误：找不到文件夹！请检查 prepare_shapenet_dataset.py 是否成功运行并在 {DATASET_NAME} 里生成了数据。")
        return

    lines = []
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                # 🟢 [修改] 使用 splitext 更安全地提取文件名 (去除后缀)
                # 原代码 filename.split('.')[-2] 在遇到文件名有点号时会出错
                model_id = os.path.splitext(filename)[0]
                lines.append(model_id)

    if len(lines) == 0:
        print("⚠️ 警告：目录是空的，没有找到 .npy 文件！")
        return

    # 排序，保证每次生成的列表顺序一致
    lines.sort()

    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lines))
        
    print(f"✅ 成功生成索引文件！")
    print(f"📝 文件名: {OUTPUT_FILE}")
    print(f"🔢 模型数量: {len(lines)}")
    print("------------------------------------------------")
    print("前 5 个模型 ID 预览:")
    for l in lines[:5]:
        print(f" - {l}")

if __name__ == '__main__':
    create_textfile()