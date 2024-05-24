import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from ada_dataloader import FrameDataset, clip_transform
from torch.utils.data import DataLoader
from ada_clip_network import AdaClipNetwork, ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter



#predict 函数中 threshold和sigma的确定!
def predict(model, img1, img2, transform, device, threshold, sigma):
    model.eval()
    with torch.no_grad():
        img1 = transform(img1).unsqueeze(0).to(device)
        img2 = transform(img2).unsqueeze(0).to(device)
        output1, output2 = model(img1, img2)
        euclidean_distance = F.pairwise_distance(output1, output2).item()


        prediction = 1 if euclidean_distance >= threshold else 0
        # 计算相似性得分
        similarity_score = torch.exp(-euclidean_distance / (2 * sigma ** 2)).item()
    return euclidean_distance,  prediction, similarity_score

def validate(model, dataloader, criterion, device, threshold):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            running_loss += loss.item()

            euclidean_distance = F.pairwise_distance(output1, output2)
            # 预测：距离小于阈值认为是正样本对（相似，标签为0），大于等于阈值认为是负样本对（不相似，标签为1）
            predictions = (euclidean_distance >= threshold).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


'''
# 验证集DataLoader（假设已经定义好）
validation_dataset = FrameDataset('path_to_validation_dir1', 'path_to_validation_dir2', transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# 执行验证
avg_loss, accuracy = validate(model, validation_dataloader, criterion, device, threshold=0.5)
print(f'Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}')
'''