import os
import torch
from ada_dataloader import FrameDataset
from torch.utils.data import DataLoader
from ada_clip_network import AdaClipNetwork, ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter
from config import config

writer = SummaryWriter('runs/adaCLIP_experiment')
# 训练循环
def train_model(model, dataloader, criterion, optimizer, num_epochs,save_path):
    model.train()
    device = config.device

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,(img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每一步记录损失
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')

        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch + 1}.pth'))

    writer.close()


