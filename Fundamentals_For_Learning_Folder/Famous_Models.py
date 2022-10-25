from Fundamentals_For_Learning import *


class LeNet(Classifier):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )


class AlexNet(Classifier):

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096),
            nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )


def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(Classifier):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        conv_blocks = []
        in_channels = 1
        for (num_convs, out_channels) in arch:
            conv_blocks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(*conv_blocks, nn.Flatten(),
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.LazyLinear(num_classes)
                                 )


def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())


class NiN(Classifier):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())


class Inception(nn.Module):
    # ci are the out_channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        # branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, X):
        b1 = F.relu(self.b1_1(X))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(X))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(X))))
        b4 = F.relu(self.b4_2(self.b4_1(X)))
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogLeNet(Classifier):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(),
                                 self.b4(), self.b5(), nn.LazyLinear(num_classes))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 32),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b5(self):
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
