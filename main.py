import torch
import ada_clip_network
import ada_clip_train
import ada_dataloader
import ada_clip_predict_validate
from torch.utils.data import DataLoader
import torch
from ada_dataloader import FrameDataset, clip_transform
from torch.utils.data import DataLoader
from ada_clip_network import AdaClipNetwork, ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter

# 设备设置和训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置目录路径和DataLoader
dir1 = 'path_to_dir1'
dir2 = 'path_to_dir2'
dataset = ada_dataloader.FrameDataset(dir1, dir2)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型和损失函数
net = ada_clip_network.AdaClipNetwork(device).to(device)
criterion = ada_clip_network.ContrastiveLoss()
optimizer = torch.optim.Adam(net.mlp.parameters(), lr=0.001)

#train model
ada_clip_train.train_model(net, dataloader, criterion, optimizer, num_epochs=25)

#predict
# 加载模型权重
model = AdaClipNetwork(device).to(device)
model.load_state_dict(torch.load('model_weights/epoch_25.pth'))
model.eval()  # 切换模型到评估模式

# transform
transform = clip_transform()
ada_clip_predict_validate.predict(model, img1, img2, transform, device, threshold, sigma)

'''使用TensorBoard

在训练过程中，可以启动TensorBoard以实时查看损失变化。打开终端，进入训练脚本所在的目录，运行以下命令：

tensorboard --logdir=runs

然后在浏览器中打开http://localhost:6006，即可看到训练过程中记录的损失曲线。
'''
