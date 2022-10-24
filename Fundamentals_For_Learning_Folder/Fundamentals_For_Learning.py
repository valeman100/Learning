import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


def accuracy(X, y, model):
    out = model(X.to(device)).argmax(axis=1).to('cpu')
    compare = (out == y).type(torch.float32)
    return compare.mean()


class GeneralDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.X = None
        self.y = None

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        return {"X": self.X[idx], 'y': self.y[idx]}


def get_model(model, example_data=(1, 1, 28, 28)):

    model = model.to(device)
    model.layer_summary(example_data)
    model.apply(init_cnn)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n Number of parameters: %d' % num_params)
    return model, nn.CrossEntropyLoss()


def data_preparation(batch=128, resize=(224, 224)):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(resize), transforms.Normalize((0.5,), (0.5,))])

    data_train = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', train=True, transform=transform, download=True)
    data_test = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', train=False, transform=transform, download=True)

    split1 = int(np.floor(0.8*len(data_train)))
    split2 = int(np.floor(0.2*len(data_train)))
    train = data_train
    test = data_test

    bs = batch
    train_dl = DataLoader(train, batch_size=bs, shuffle=False, sampler=range(split1), num_workers=2, pin_memory=True)
    val_dl = DataLoader(train, batch_size=bs, shuffle=False, sampler=range(split2), num_workers=2, pin_memory=True)
    test_dl = DataLoader(test, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)

    print(len(train), len(test), len(train_dl), len(val_dl), len(test_dl))

    return train_dl, val_dl, test_dl


class LeNet(nn.Module):
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

    def forward(self, X):
        return self.net(X)

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape).to(device)
        print('X', 'output shape:\t', X.shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class AlexNet(LeNet):

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


def fit(train_dl, val_dl, test_dl, loss_f, model, lr=0.1, epochs=15):

    opt = optim.SGD(model.parameters(), lr=lr)
    train_loss, val_loss, acc = [], [], []

    for epoch in range(epochs):

        epoch_loss, iteration = 0, 0
        model.train()
        for i, (X, y) in enumerate(train_dl):
            out = model(X.to(device))
            loss = loss_f(out, y.to(device))

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            iteration += 1
        train_loss.append(epoch_loss / iteration)

        epoch_loss, iteration = 0, 0
        model.eval()
        for X, y in val_dl:
            with torch.no_grad():
                out = model(X.to(device))
                epoch_loss += loss_f(out, y.to(device)).item()
                iteration += 1
        val_loss.append(epoch_loss / iteration)

        epoch_loss, iteration = 0, 0
        for X, y in test_dl:
            with torch.no_grad():
                epoch_loss += accuracy(X, y, model)
                iteration += 1
        acc.append(epoch_loss / iteration)

        print("Epoch: {}, loss = {}, v_loss = {}, accuracy = {}".format(epoch, train_loss[-1], val_loss[-1], acc[-1]))

    return train_loss, val_loss, acc







