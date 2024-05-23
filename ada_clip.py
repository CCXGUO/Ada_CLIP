import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# 加载预训练的 CLIP 模型和相应的预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 冻结视觉编码器参数
for param in model.visual.parameters():
    param.requires_grad = False

# 准备要处理的图像
url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# 预处理图像
image_input = preprocess(img).unsqueeze(0).to(device)

# 提取视觉嵌入
with torch.no_grad():
    image_features = model.encode_image(image_input)

# 打印提取的视觉嵌入
print("Visual Embeddings:", image_features)

# 可视化输入图像
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')
plt.show()
