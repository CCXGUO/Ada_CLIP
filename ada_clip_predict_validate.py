import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import torch
from PIL import Image
import clip
from ada_clip_network import AdaClipNetwork
from config import config


def preprocess_image(image_path, preprocess):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)

def predict(model, img1_path, img2_path, device, threshold):
    model.to(device)
    model.eval()

    # 加载 CLIP 的预处理函数
    _, preprocess = clip.load("ViT-B/32", device=device)

    # 对图像进行预处理
    img1 = preprocess_image(img1_path, preprocess).to(device)
    img2 = preprocess_image(img2_path, preprocess).to(device)

    # 进行预测
    with torch.no_grad():
        output1, output2 = model.inference(img1, img2)
        euclidean_distance = F.pairwise_distance(output1, output2).item()
        prediction = 1 if euclidean_distance >= threshold else 0

    return euclidean_distance, prediction

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


if __name__ == "__main__":
    img1_path = "/home/luno/dev/Adaclip_data/validation_threshold/Video1/frame_53.png"
    img2_path = "/home/luno/dev/Adaclip_data/validation_threshold/Video2/frame_53.png"
    model_path = "/home/luno/dev/Ada_CLIP/model_weights/epoch_25.pth"
    threshold = 0.5  # 根据需求设置阈值

    device = config.device

    model = AdaClipNetwork(device=device)
    model.load_state_dict(torch.load(model_path))

    euclidean_distance, prediction = predict(model, img1_path, img2_path, device, threshold)
    print(f"Euclidean Distance: {euclidean_distance}")
    print(f"Prediction: {prediction}")