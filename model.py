import torch.nn as nn
import torch.nn.functional as F
import torch

class FoodClassifier(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super(FoodClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.flat_size = self._calculate_flat_size(img_size)
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def _calculate_flat_size(self, img_size):
        x = torch.zeros(1, 3, img_size, img_size)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
