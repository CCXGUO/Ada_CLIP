import os
import torch
from torch.utils.tensorboard import SummaryWriter
from config import config
from tqdm import tqdm
import ada_clip_network
import ada_dataloader
from torch.utils.data import DataLoader


writer = SummaryWriter('runs/adaCLIP_experiment')
# 训练循环
def train_model(model, dataloader, criterion, optimizer, num_epochs,save_path):
    model.train()
    device = config.device

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,(img1, img2, label) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            img1, img2, label = img1.to(device).float(), img2.to(device).float(), label.to(device).float()

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

if __name__ == "__main__":
    # 设备设置和训练
    device = config.device
    save_path = config.weights_save_path
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    lr = config.learning_rate

    # 设置目录路径和DataLoader
    dir1 = config.dir1
    dir2 = config.dir2
    dataset = ada_dataloader.FrameDataset(dir1, dir2)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # 初始化模型和损失函数
    net = ada_clip_network.AdaClipNetwork(device = config.device).to(device)
    criterion = ada_clip_network.ContrastiveLoss()
    optimizer = torch.optim.Adam(net.mlp.parameters(), lr)

    # train model_weights
    train_model(net, dataloader, criterion, optimizer, num_epochs, save_path)
