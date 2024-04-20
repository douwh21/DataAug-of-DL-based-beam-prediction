import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_3D(nn.Module):

    def __init__(self, N=60, K=32, Tx=8, Channel=2):
        super(Model_3D, self).__init__()

        # first dimension in CNN denotes beam training number
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64,
                               kernel_size=(1,3), stride=(1,3), padding=(0,1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256,
                               kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, 64)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):

        x = self.bn0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        P_dim_size = x.shape[3]
        x = nn.MaxPool2d(kernel_size=(1, P_dim_size))(x)
        x = torch.squeeze(x)

        x = x.permute(2, 0, 1)

        y1 = self.drop(x)
        y1 = self.fc(y1)

        result = y1

        return result
