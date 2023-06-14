import torch
import torch.nn as nn
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()

        self.model = models.resnet18(pretrained=False)
        self.fc1 = nn.Linear(512, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
