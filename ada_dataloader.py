import os
import random
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# 定义函数加载帧
def load_frames_from_dir(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            frame = Image.open(filepath)
            frames.append((filename, frame))
    return frames


# 定义计算最近邻帧的函数
def find_nearest_neighbors(frames, frame, num_neighbors):
    # 使用某种特征提取方法，比如像素值展平
    frame_features = [np.array(f[1]).flatten() for f in frames]
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(frame_features)
    frame_feature = np.array(frame).flatten().reshape(1, -1)
    distances, indices = nbrs.kneighbors(frame_feature)
    return indices[0]


# 自定义数据集类
class FramePairsDataset(Dataset):
    def __init__(self, dir1, dir2, transform=None):
        self.dir1_frames = load_frames_from_dir(dir1)
        self.dir2_frames = load_frames_from_dir(dir2)
        self.transform = transform
        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        total_frames = len(self.dir1_frames)
        num_neighbors = max(1, int(total_frames * 0.05))

        for i, (filenameA, frameA) in enumerate(self.dir1_frames):
            setA_indices = find_nearest_neighbors(self.dir1_frames, frameA, num_neighbors)
            frameB = self.dir2_frames[i][1]
            setB_indices = find_nearest_neighbors(self.dir2_frames, frameB, num_neighbors)
            setA = [self.dir1_frames[idx][1] for idx in setA_indices]
            setB = [self.dir2_frames[idx][1] for idx in setB_indices]

            # 创建正样本对
            positive_pair = (frameA, random.choice(setA + setB))
            pairs.append((positive_pair, 1))  # 1 表示正样本

            # 创建负样本对
            all_frames = self.dir1_frames + self.dir2_frames
            setA_setB_indices = set(setA_indices).union(setB_indices)
            negative_candidates = [all_frames[idx][1] for idx in range(total_frames * 2) if
                                   idx not in setA_setB_indices]
            negative_pair = (frameA, random.choice(negative_candidates))
            pairs.append((negative_pair, 0))  # 0 表示负样本

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (frameA, frameB), label = self.pairs[idx]
        if self.transform:
            frameA = self.transform(frameA)
            frameB = self.transform(frameB)
        return (frameA, frameB), label


# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
dataset = FramePairsDataset('dir1', 'dir2', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
