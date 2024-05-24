import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class AdaClipNetwork(nn.Module):
    def __init__(self,device):
        super(AdaClipNetwork, self).__init__()
        self.device = device
        self.clip_model = clip.load("ViT-B/32",device= device)
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_once(self, image):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(image)
        output = self.mlp(image_features)
        return output

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive