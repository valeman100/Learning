import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GeneralDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.X = None
        self.y = None

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        return {"X": self.X[idx], 'y': self.y[idx]}


def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)


def layer_summary(model, X_shape):
    X = torch.randn(*X_shape).to(device)
    print('X', 'output shape:\t', X.shape)
    for layer in model.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)


def get_model(model, example_data=(1, 1, 28, 28)):
    print("\n1")
    model = model.to(device)
    print("\n1")
    layer_summary(model, example_data)
    print("\n1")
    model.apply(init_cnn)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNumber of parameters: {num_params:d}')
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


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = None

    def forward(self, X):
        return self.net(X)




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

        print("Epoch: {}, loss = {:.6}, v_loss = {:.6}, accuracy = {:.6}".format(epoch, train_loss[-1], val_loss[-1], acc[-1]))

    return train_loss, val_loss, acc


def after_training_plots(train_loss, val_loss, acc):

    plt.plot(train_loss, label='training')
    plt.plot(val_loss, label='validation')
    plt.plot(acc, label='accuracy')

    plt.legend()
    plt.title('Losses')
    plt.xlabel("Epochs")
    plt.show()


def accuracy(X, y, model):
    out = model(X.to(device)).argmax(axis=1).to('cpu')
    compare = (out == y).type(torch.float32)
    return compare.mean()


def model_test(X, y, model):

    # X, y = next(iter(test_dl))
    model.cuda()
    out = model(X.cuda())
    comparison = out.max(dim=1)[1].cpu() == y

    print('prediction comparison:\n\n', comparison, '\n\nAccuracy = {}'.format(accuracy(X, y, model)))





