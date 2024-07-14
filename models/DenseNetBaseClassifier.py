import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models import densenet201, DenseNet201_Weights
from torchsummary import summary
from typing import Literal


class DenseNetBaseClassifier(nn.Module):
    def __init__(self,
                 n_classes: int,
                 weights_version: Literal[0, 1]=1) -> None:
        """"""
        super().__init__()
        self.n_classes = n_classes

        if weights_version == 0:
            self.model = densenet201(weights=DenseNet201_Weights.DEFAULT)
        elif weights_version == 1:
            self.model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(1920, 1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        print(self.model)
        summary(self.model, (3, 224, 224))


    def get_fc_parameters(self) -> Parameter:
        return self.model.classifier.parameters()