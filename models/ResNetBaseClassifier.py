import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
from typing import Literal

class ResNetBaseClassifier(nn.Module):
    def __init__(self,
                 n_classes: int,
                 weights_version: Literal[1, 2]=1) -> None:
        """"""
        super().__init__()
        self.n_classes = n_classes

        if weights_version == 1:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif weights_version == 2:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in self.model.parameters():
            param.requires_grad = False

        # inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, self.n_classes)
        )


    def forward(self,
                inputs: torch.Tensor) -> torch.Tensor:
        """"""
        output = self.model(inputs)
        output = F.softmax(output, dim=1)

        return output


    def summary(self) -> None:
        """"""
        summary(self.model, (3, 224, 224))


    def get_fc_parameters(self) -> Parameter:
        return self.model.fc.parameters()