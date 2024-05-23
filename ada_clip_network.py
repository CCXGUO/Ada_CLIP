import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)


# get CLIP embeddings
def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")


import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import ClipDataset, dataset_collate

# get visual features
def get_image_feature(datasets_path, data_json_path, batch_size, num_workers, model):
    # 计算样本数
    val_lines = json.load(open(data_json_path, mode='r', encoding='utf-8'))
    num_val = len(val_lines)
    # 创建验证数据集
    val_dataset = ClipDataset([model.config['input_resolution'], model.config['input_resolution']], val_lines,
                              datasets_path, random=False)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=False, collate_fn=dataset_collate, sampler=None)

    # 获得视觉特征和文本特征
    # 避免计算梯度，因为这里只是在进行评估
    i_features = []
    i_label = []
    image_names = []
    for iteration, batch in tqdm(enumerate(gen_val)):
        images, texts, images_name = batch
        with torch.no_grad():
            if model.cuda:
                images = images.cuda()

            images_feature, _ = model.detect_image_for_eval(images, texts=None)
            i_features.append(images_feature)
            i_label.extend(texts)
            image_names.extend(images_name)

    i_features = torch.cat(i_features, 0) #是否应该合并为一个大的张量?
    i_features = i_features / i_features.norm(dim=-1, keepdim=True)
    return i_features, i_label, image_names