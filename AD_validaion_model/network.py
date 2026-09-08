import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 2
class BaselineCNN(nn.Module):
    # Basic CNN using 2D convolutional layers, some max pooling and
    # a single batch normalisation layer to counter overfitting. We used a linear (dense)
    # layer as the output with the output shape being the number of classes
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
class BaselineCNN_Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*32*32, 128)
        self.out = nn.Linear(128, N_CLASSES)

    def forward_with_features(self, x):
        features = []

        # 卷积层1
        x = F.relu(self.conv1(x))
        features.append(x)

        x = self.pool1(x)
        x = self.batchnorm1(x)

        # 卷积层2
        x = F.relu(self.conv2(x))
        features.append(x)

        x = self.pool2(x)
        x = self.flatten(x)

        # 全连接层
        x = F.relu(self.fc1(x))
        features.append(x)

        out = self.out(x)
        return out, features
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class EightLayerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(EightLayerCNN, self).__init__()

        # 特征提取部分 (8层卷积)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Layer 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Layer 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Layer 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Layer 5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Layer 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Layer 7
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Layer 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8 -> 4
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),  # FC 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)  # FC 2 (Output)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TwelveLayerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(TwelveLayerCNN, self).__init__()

        # 输入: [1, 128, 128]
        self.features = nn.Sequential(
            # --- Block 1: 浅层特征 (边缘/纹理) ---
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            # --- Block 2: 基础形状 ---
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # --- Block 3: 局部部件 ---
            # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # --- Block 4: 抽象语义 (中间层) ---
            # Layer 7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Layer 8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # --- Block 5: 高级语义 ---
            # Layer 9
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Layer 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8 -> 4

            # --- Block 6: 瓶颈层 (Bottleneck) ---
            # Layer 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Layer 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 4 -> 2
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 输入尺寸: 512通道 * 2高 * 2宽 = 2048
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

