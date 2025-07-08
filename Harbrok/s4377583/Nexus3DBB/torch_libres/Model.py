import torch
import torch.nn as nn
import torch.nn.functional as F


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    batch_size = orient_batch.size(0)
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch   = orient_batch[torch.arange(batch_size), indexes]

    theta_diff           = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1],   orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()


class Model(nn.Module):
    def __init__(self, features=None, bins=2, w=0.4):
        """
        features: nn.Sequential of conv layers (e.g. resnet18 up to layer4)
        """
        super(Model, self).__init__()
        self.bins     = bins
        self.w        = w
        self.features = features

        # ensure a 7x7 output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        fc_input_dim = 512 * 7 * 7

        # orientation head: outputs bins×2 (sin, cos)
        self.orientation = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins * 2),
        )

        # confidence head: outputs bins logits
        self.confidence = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
        )

        # dimension head: outputs 3D dims
        self.dimension = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        # backbone convs
        x = self.features(x)             
        # force to 7×7
        x = self.adaptive_pool(x)        
        # flatten
        x = x.view(x.size(0), -1)        

        # orientation: [B, bins, 2]
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)

        # confidence: [B, bins]
        confidence = self.confidence(x)

        # dimensions: [B, 3]
        dimension = self.dimension(x)

        return orientation, confidence, dimension

