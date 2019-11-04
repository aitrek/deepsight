"""Convolutional Recurrent Neural Network"""
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, n_cls: int):
        super().__init__()
        # Convolutional Layers
        self._n_cls = n_cls
        self.cnn = nn.Sequential()
        self.cnn.add_module("conv0", nn.Conv2d(in_channels=1,    # gray scale image
                                               out_channels=64,
                                               kernel_size=(3, 3),
                                               stride=1,
                                               padding=1))
        self.cnn.add_module("relu0", nn.ReLU(inplace=True))
        self.cnn.add_module("max_pooling0", nn.MaxPool2d(kernel_size=(2, 2),
                                                         stride=2))

        self.cnn.add_module("conv1", nn.Conv2d(in_channels=64,
                                               out_channels=128,
                                               kernel_size=(3, 3),
                                               stride=1,
                                               padding=1))
        self.cnn.add_module("relu1", nn.ReLU(inplace=True))
        self.cnn.add_module("max_pooling1", nn.MaxPool2d(kernel_size=(2, 2),
                                                         stride=2))

        self.cnn.add_module("conv2", nn.Conv2d(in_channels=128,
                                               out_channels=256,
                                               kernel_size=(3, 3),
                                               stride=1,
                                               padding=1))
        self.cnn.add_module("bn2", nn.BatchNorm2d(num_features=256))
        self.cnn.add_module("relu2", nn.ReLU(inplace=True))

        self.cnn.add_module("conv3", nn.Conv2d(in_channels=256,
                                               out_channels=256,
                                               kernel_size=(3, 3),
                                               stride=1,
                                               padding=1))
        self.cnn.add_module("relu3", nn.ReLU(inplace=True))
        self.cnn.add_module("max_pooling3", nn.MaxPool2d(kernel_size=(2, 2),
                                                         stride=(2, 1),
                                                         padding=(0, 1)))

        self.cnn.add_module("conv4", nn.Conv2d(in_channels=256,
                                               out_channels=512,
                                               kernel_size=(3, 3),
                                               stride=1,
                                               padding=1))
        self.cnn.add_module("bn4", nn.BatchNorm2d(num_features=512))
        self.cnn.add_module("relu4", nn.ReLU(inplace=True))

        self.cnn.add_module("conv5", nn.Conv2d(in_channels=512,
                                               out_channels=512,
                                               kernel_size=(3, 3),
                                               stride=1,
                                               padding=1))
        self.cnn.add_module("relu5", nn.ReLU(inplace=True))
        self.cnn.add_module("max_pooling5", nn.MaxPool2d(kernel_size=(2, 2),
                                                         stride=(2, 1),
                                                         padding=(0, 1)))

        self.cnn.add_module("conv6", nn.Conv2d(in_channels=512,
                                               out_channels=512,
                                               kernel_size=(2, 2),
                                               stride=1,
                                               padding=0))
        self.cnn.add_module("bn6", nn.BatchNorm2d(num_features=512))
        self.cnn.add_module("relu6", nn.ReLU(inplace=True))

        # Recurrent Layers
        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=256,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)

        # Linear Layer
        self.linear = nn.Linear(512, self._n_cls)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2).permute(2, 0, 1)
        self.rnn.flatten_parameters()
        x = self.rnn(x)[0]
        T, B, C = x.shape
        x = x.reshape(-1, C)
        x = self.linear(x)
        x = x.reshape(T, B, -1)
        x = x.transpose(0, 1)
        return x
