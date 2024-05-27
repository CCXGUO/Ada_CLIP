import numpy as np
import clip
import torch.nn.functional as F
import torch
import ada_dataloader
import ada_clip_network
from config import config
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
def evaluate_thresholds(model, dataloader, device, thresholds):
    best_threshold = None
    best_f1 = 0
    results = []

    for threshold in thresholds:
        all_predictions = []
        all_labels = []
        for batch in dataloader:
            img1, img2, labels = batch
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # 打印批处理图像尺寸
            print(f"Batch img1 shape: {img1.shape}, Batch img2 shape: {img2.shape}")

            for i in range(len(labels)):
                _, prediction = predict(model, img1[i], img2[i], device, threshold)
                all_predictions.append(prediction)
                all_labels.append(labels[i].item())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        results.append((threshold, accuracy, precision, recall, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, results


dir1 = '/home/luno/dev/Ada_CLIP/validation_threshold/Video2'
dir2 = '/home/luno/dev/Ada_CLIP/validation_threshold/Video1'
batch_size = config.batch_size
dataset = ada_dataloader.FrameDataset(dir1, dir2)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

thresholds = np.arange(0, 2.1, 0.1)

# Initialize the model and other parameters
device = config.device
model =  ada_clip_network.AdaClipNetwork(device)

# 加载保存的模型权重
model_path = '/home/luno/dev/Ada_CLIP/model_weights/epoch_25.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# 设置模型为评估模式
model.eval()



best_threshold, results = evaluate_thresholds(model, dataloader, device, thresholds)

# Print the results
for threshold, accuracy, precision, recall, f1 in results:
    print(
        f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

print(f"Best threshold: {best_threshold:.2f}")





