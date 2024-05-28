import torch
import ada_clip_network
import ada_clip_train
import ada_dataloader
import ada_clip_predict_validate
from torch.utils.data import DataLoader
import torch
from ada_dataloader import FrameDataset
from torch.utils.data import DataLoader
from ada_clip_network import AdaClipNetwork, ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter
from config import config

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
net = ada_clip_network.AdaClipNetwork(device).to(device)
criterion = ada_clip_network.ContrastiveLoss()
optimizer = torch.optim.Adam(net.mlp.parameters(), lr)

#train model_weights
ada_clip_train.train_model(net, dataloader, criterion, optimizer, num_epochs, save_path)
#
# #predict
# # 加载模型权重
# model_weights = AdaClipNetwork(device).to(device)
# model_weights.load_state_dict(torch.load('model_weights/epoch_25.pth'))
# model_weights.eval()  # 切换模型到评估模式

# transform
# transform = clip_transform()
# ada_clip_predict_validate.predict(model_weights, img1, img2, transform, device, threshold, sigma)

'''使用TensorBoard

在训练过程中，可以启动TensorBoard以实时查看损失变化。打开终端，进入训练脚本所在的目录，运行以下命令：

tensorboard --logdir=runs

然后在浏览器中打开http://localhost:6006，即可看到训练过程中记录的损失曲线。
'''
