import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import Tuple, List
from ..utils.datasets import AnchorData


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
        # todo the transposes and cat below right?
        # todo no need transpose for lstm?
        x = x.transpose(1, 3)
        x = x.transpose(1, 2)
        x = torch.cat([t for t in x], dim=0)
        return x


class CTPN(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = models.vgg16(pretrained=True).features[:-1]
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.rnn = nn.Sequential()
        self.rnn.add_module("bsu", SpatialUnfold(3, 1, 1))
        self.rnn.add_module("blstm", nn.LSTM(input_size=3 * 3 * 512,
                                             hidden_size=128,
                                             num_layers=1,
                                             batch_first=True,
                                             bidirectional=True))

        self.fc = nn.Sequential()
        # todo enough with only one fc layer?
        self.fc.add_module("fc", nn.Conv2d(in_channels=256,
                                           out_channels=512,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0))
        self.fc.add_module("relu0", nn.ReLU(inplace=True))
        self.fc.add_module("fc1", nn.Conv2d(in_channels=512,
                                            out_channels=(2 + 2 + 1) * 10,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.fc(x)
        return x


class CTPNLoss(nn.Module):

    def __init__(self, Ns: int = 128, k: int = 10):
        super().__init__()
        self._Ns = Ns       # number of anchors for each mini-batch
        self._k = k
        self._use_cuda = False
        self._ratio = 0.5   # ratio for positive and negative samples.
        self._lambda1 = 1.0
        self._lambda2 = 2.0
        self._Ls = nn.CrossEntropyLoss()    # text/non-text
        self._Lv = nn.SmoothL1Loss()        # coordinate
        self._Lo = nn.SmoothL1Loss()        # side-refinement

    def cuda(self, device=None):
        super().cuda(device)
        self._use_cuda = True

    def forward(self, x: torch.Tensor, anchor_data: AnchorData):
        def get_choices(h: int, w: int, k: int) -> List[Tuple[int, int, int]]:
            idx_lst = list(range(h * w * k))
            random.shuffle(idx_lst)
            chs = []
            for idx in idx_lst:
                yi = idx // (w * k)
                xi = idx % (w * k) // k
                zi = idx % (w * k) % k
                chs.append((yi, xi, zi))
            return chs

        h, w, k = anchor_data.shape
        choices = get_choices(h, w, k)

        n_pos = self._Ns * self._ratio
        n_neg = self._Ns - n_pos

        pos = []
        neg = []
        for yi, xi, zi in choices:
            data = anchor_data.get(yi, xi, zi)
            if data[0] == 1:
                pos.append((yi, xi, zi))
            elif data[0] == -1:
                neg.append((yi, xi, zi))
            else:
                continue

            if len(pos) > n_pos and len(neg) > n_neg:
                break

        pos = pos if len(pos) <= n_pos else pos[:n_pos]
        neg = neg[:self._Ns-len(pos)]   # the length of neg is supposed much larger than self._Ns

        loss_cls = []
        loss_reg_v = []
        loss_reg_o = []

        if self._use_cuda:
            for yi, xi, zi in pos:
                data = anchor_data.get(yi, xi, zi)
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[0, yi, xi, start:end],
                                         torch.tensor([1]).cuda()))
                # vertical coordinates
                start = zi * 2
                end = start + 2
                loss_reg_v.append(self._Lv(x[0, yi, xi, start:end],
                                           torch.tensor(data[2:4]).cuda()))
                # side-refinement
                if data[1] != 0:
                    start = self._k * 4 + zi
                    end = start + 1
                    loss_reg_o.append(self._Lo(x[0, yi, xi, start:end],
                                               torch.tensor(data[-2]).cuda()))

            for yi, xi, zi in neg:
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[0, yi, xi, start:end],
                                         torch.tensor([0]).cuda()))
        else:
            for yi, xi, zi in pos:
                data = anchor_data.get(yi, xi, zi)
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[0, yi, xi, start:end],
                                         torch.tensor([1])))
                # vertical coordinates
                start = zi * 2
                end = start + 2
                loss_reg_v.append(self._Lv(x[0, yi, xi, start:end],
                                           torch.tensor(data[2:4])))
                # side-refinement
                if data[1] != 0:
                    start = self._k * 4 + zi
                    end = start + 1
                    loss_reg_o.append(self._Lo(x[0, yi, xi, start:end],
                                               torch.tensor(data[-2])))

            for yi, xi, zi in neg:
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[0, yi, xi, start:end],
                                         torch.tensor([0])))

        loss = sum(loss_cls) / len(loss_cls) + sum(loss_reg_v) / len(loss_reg_o)
        if len(loss_reg_o) > 0:
            loss += sum(loss_reg_o) / len(loss_reg_o)

        return loss
