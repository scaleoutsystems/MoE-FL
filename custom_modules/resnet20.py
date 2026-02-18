import torch.nn.functional as F
import torch.nn as nn

#Simple CIFAR-10 style ResNet-20 implementation
class BasicBlock(nn.Module):
        def __init__(self, in_c, out_c, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_c)
            self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_c)

            self.proj = None
            if stride != 1 or in_c != out_c:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c),
                )

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            x = x if self.proj is None else self.proj(x)
            return F.relu(x + y)

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_c = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, blocks=3, stride=1)
        self.layer2 = self._make_layer(32, blocks=3, stride=2)
        self.layer3 = self._make_layer(64, blocks=3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(64, num_classes)

    def _make_layer(self, out_c, blocks, stride):
        layers = [BasicBlock(self.in_c, out_c, stride)]
        self.in_c = out_c
        for _ in range(blocks - 1):
            layers.append(BasicBlock(self.in_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)