import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ImageToPosNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        # self.zer = torch.ones((4,4))

    def _vec2ss_matrix(self, vector):  # vector to skewsym. matrix

        ss_matrix = torch.zeros((3,3))
        ss_matrix[0, 1] = -vector[2]
        ss_matrix[0, 2] = vector[1]
        ss_matrix[1, 0] = vector[2]
        ss_matrix[1, 2] = -vector[0]
        ss_matrix[2, 0] = -vector[1]
        ss_matrix[2, 1] = vector[0]

        return ss_matrix
    
    def camera_transform(self, x: torch.Tensor):

        if len(x.shape) == 2:
            res = torch.zeros((x.shape[0], 4, 4)).to(x.get_device())
            for batch_idx in range(x.shape[0]):
                res[batch_idx] = self.camera_transform(x[batch_idx])
            return res

        w = x[:3]
        v = x[3:6]
        theta = x[6]

        exp_i = torch.zeros((4,4))
        w_skewsym = self._vec2ss_matrix(w)
        exp_i[:3, :3] = torch.eye(3) + torch.sin(theta) * w_skewsym + (1 - torch.cos(theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3) * theta + (1 - torch.cos(theta)) * w_skewsym + (theta - torch.sin(theta)) * torch.matmul(w_skewsym, w_skewsym), v)
        exp_i[3, 3] = 1.

        return exp_i

    def forward(self, x):
        
        x = self.conv0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # return x
        # pose = self.camera_transform(x)
        return x.reshape(-1,4,4)