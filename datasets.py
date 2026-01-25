import torch
from torch.utils.data import Dataset
import os
import numpy as np

class VoxelDataset(Dataset):
    def __init__(self, files, clamp=0.1, rescale_sdf=True):
        self.files = files
        self.clamp = clamp
        self.rescale_sdf = rescale_sdf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        array = np.load(self.files[index])
        result = torch.from_numpy(array)
        if self.clamp is not None:
            result.clamp_(-self.clamp, self.clamp)
            if self.rescale_sdf:
                result /= self.clamp
        return result

    @staticmethod
    def from_split(pattern, split_file_name):
        with open(split_file_name, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        files = [pattern.format(id) for id in ids]
        files = [file for file in files if os.path.exists(file)]
        return VoxelDataset(files)

# ==========================================
# 👇 核心修改区域：PointDataset
# ==========================================
class PointDataset(Dataset):
    def __init__(self, root, filenames, num_points=1024, transform=None):
        self.root = root
        self.filenames = filenames
        self.num_points = num_points
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # 🟢 [修改 1] 路径构建更加稳健
        # 你的数据在 data/shapenet/00_good_selected/uniform/ 下
        # 这里确保路径拼接正确
        uniform_path = os.path.join(self.root, 'uniform', f'{name}.npy')
        
        # 🟢 [修改 2] 增加文件存在性检查 (防止训练中断)
        if not os.path.exists(uniform_path):
            raise FileNotFoundError(f"❌ 找不到数据文件: {uniform_path}")

        # 加载数据 (强制转为 float32，防止 PyTorch 类型报错)
        points_all = np.load(uniform_path).astype(np.float32)

        # 🟢 [修改 3] 智能采样算法优化
        # 原代码默认 replace=True (允许重复抽样)，这在点数足够时不科学。
        # 改进：如果点数够，就用 replace=False (不重复，覆盖更广)；如果点数不够，才允许重复。
        total_points = points_all.shape[0]
        if total_points >= self.num_points:
            sample_indices = np.random.choice(total_points, self.num_points, replace=False)
        else:
            sample_indices = np.random.choice(total_points, self.num_points, replace=True)
            
        uniform = points_all[sample_indices]
        
        # 转为 Tensor
        data = torch.from_numpy(uniform)

        if self.transform is not None:
            data = self.transform(data)

        return data

    @staticmethod
    def from_split(root, split, num_points=1024, transform=None):
        # 🟢 [修改 4] 更安全的 train.txt 读取方式
        # 确保去掉换行符，并且忽略空行
        split_file = os.path.join(root, f'{split}.txt')
        
        if not os.path.exists(split_file):
             raise FileNotFoundError(f"❌ 找不到索引文件: {split_file}\n请检查是否已将 train.txt 移动到 {root} 目录下！")

        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
            
        print(f"✅ 成功加载数据集索引: {len(filenames)} 个模型")
        return PointDataset(root, filenames, num_points, transform)

if __name__ == '__main__':
    pass