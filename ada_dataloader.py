import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
class FrameDataset(Dataset):
    def __init__(self, dir1, dir2, transform=None):
        self.dir1 = dir1
        self.dir2 = dir2
        self.transform = transform if transform is not None else clip_transform(224)
        self.frames1 = sorted([f for f in os.listdir(dir1) if f.endswith('.png')])
        self.frames2 = sorted([f for f in os.listdir(dir2) if f.endswith('.png')])
        self.total_frames = len(self.frames1)
        self.pairs = self.generate_pairs()



    def generate_pairs(self):
        pairs = []
        # set A: 5% of the total frames nearby selected frame in dir1
        # Set B: 5% of the total frames nearby selected frame in dir2
        for i, frame in enumerate(self.frames1):
            start= max(0, i - self.total_frames // 20)
            end = min(self.total_frames, i + self.total_frames // 20 + 1)
            setA = set(self.frames1[start:end])
            setB = set(self.frames2[start:end])

        # positive pairs
            positive_pool = list(setA.union(setB))
            positive_frame = random.choice(positive_pool)
            if positive_frame in setA:
                pairs.append((os.path.join(self.dir1, frame), os.path.join(self.dir1, positive_frame), 0))
            else:
                pairs.append((os.path.join(self.dir1, frame), os.path.join(self.dir2, positive_frame), 0))

        # negative pairs
            negative_pool1 = set(self.frames1) - setA
            negative_pool2 = set(self.frames2) - setB
            negative_pool = list(negative_pool1.union(negative_pool2))
            negative_frame = random.choice(negative_pool)
            if positive_frame in negative_pool1:
                pairs.append((os.path.join(self.dir1, frame), os.path.join(self.dir1, negative_frame), 1))
            else:
                pairs.append((os.path.join(self.dir1, frame), os.path.join(self.dir2, negative_frame), 1))


        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frameA_path, frameB_path, label = self.pairs[idx]
        frameA = Image.open(frameA_path).convert('RGB')
        frameB = Image.open(frameB_path).convert('RGB')

        if self.transform:
            frameA = self.transform(frameA)
            frameB = self.transform(frameB)

        return frameA, frameB, label

# 设置目录路径
dir1 = 'path_to_dir1'
dir2 = 'path_to_dir2'

# 初始化数据集和DataLoader
dataset = FrameDataset(dir1, dir2, transform=None)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 示例：迭代DataLoader
for frameA, frameB, label in dataloader:
    print(frameA.size(), frameB.size(), label)
    break
