"""Connectionist Text Proposal Network"""
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import Tuple
from .transform import ShortSideTransform


ctpn_transformer = transforms.Compose(
    [
        ShortSideTransform(600),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


def calc_anchor_heights(k: int = 10, fixed_width: int = 16):
    """
    Anchor heights:
    1. fixed_width / 0.7 ** -1, i.e. 70% fixed_width
    2. fixed_width / 0.7 ** 0， i.e. fixed_width
    3. fixed_width / 0.7 ** 1
    4. fixed_width / 0.7 ** 2
    ...
    10. fixed_width / 0.7 ** 8
    """
    r = 0.7
    s = -1
    return [round(fixed_width / r ** e) for e in range(s, k + s)]


ANCHOR_HEIGHTS = calc_anchor_heights()


class SpatialUnfold(nn.Module):
    """Unfold spatial features map to create feature vectors for lstm inputs.

    Args:
        kernel_size (int or Tuple[int, int]): Size of the unfolding kernel.
        stride (int or Tuple[int, int]): Stride of the unfolding.
        padding (int or Tuple[int, int]): Zero-padding added to both sides of
            the features maps.
    """
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        for param in [kernel_size, stride, padding]:
            assert isinstance(param, int) or \
                   (isinstance(param, tuple) and len(param) == 2)

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) \
            else (kernel_size, kernel_size)

        self._stride = stride if isinstance(stride, tuple) \
            else (stride, stride)

        self._padding = padding if isinstance(padding, tuple) \
            else (padding, padding)

    def forward(self, x):
        original_height = x.shape[2]
        x = F.unfold(x,
                     kernel_size=self._kernel_size,
                     padding=self._padding,
                     stride=self._stride)
        x = x.reshape((x.shape[0], x.shape[1], original_height, -1))
        x = x.squeeze(0).transpose(0, 2)

        return x


class CTPN(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = models.vgg16(pretrained=True).features[:-1]
        for name, param in self.cnn.named_parameters():
            if name in ["0.weight", "0.bias", "2.weight", "2.bias"]:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.rnn = nn.Sequential()
        self.rnn.add_module("bsu", SpatialUnfold(3, 1, 1))
        self.rnn.add_module("blstm", nn.LSTM(input_size=3 * 3 * 512,
                                             hidden_size=128,
                                             num_layers=1,
                                             batch_first=True,
                                             bidirectional=True))

        self.fc = nn.Sequential()
        self.fc.add_module("fc", nn.Conv2d(in_channels=256,
                                           out_channels=512,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0))
        self.vc = nn.Conv2d(in_channels=512,
                            out_channels=2 * 10,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.score = nn.Conv2d(in_channels=512,
                               out_channels=2 * 10,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.side = nn.Conv2d(in_channels=512,
                              out_channels=10,
                              kernel_size=1,
                              stride=1,
                              padding=0)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)[0]
        x = x.transpose(0, 2).unsqueeze(0)
        x = self.fc(x)
        x = F.relu(x, inplace=True)

        vcoords = self.vc(x)
        scores = self.score(x)
        sides = self.side(x)

        return vcoords, scores, sides
