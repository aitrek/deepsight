import random
import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from typing import Tuple, List
from ..utils.transforms import ShortSideTransform


ctpn_transformer = transforms.Compose(
    [
        ShortSideTransform(600),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


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
        x = self.rnn(x)[0]
        x = x.transpose(0, 2).transpose(1, 2).unsqueeze(0)
        x = self.fc(x)
        x = x.transpose(1, 3).transpose(1, 2)

        return x


class AnchorData:

    _memory = {}

    def __init__(self, index: int, anchors_list: list,
                 height: int, width: int, k: int = 10,
                 fixed_width: int = 16, memoize: bool = True):
        self._index = index
        self._height = height
        self._width = width
        self._k = k
        self._fixed_width = fixed_width
        self._memoize = memoize
        self.data = self._get_data(anchors_list)

    def _get_anchor_heights(self):
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
        return [round(self._fixed_width / r ** e) for e in range(s, self._k + s)]

    def _get_data(self, anchors_list: list):
        if self._memoize:
            if self._index not in self._memory:
                data = self._get_data_(anchors_list)
                self._memory[self._index] = data
            return self._memory[self._index]
        else:
            return self._get_data_(anchors_list)

    def _get_data_(self, anchors_list: list):

        def cal_iou(cy: int, h: int, cy_gt: float, h_gt: int) -> float:
            """
            Calculate iou.

            Args:
                cy (int): Center of the anchor box on y-axis.
                h (int): Height of the anchor.
                cy_gt (int): Center of the GT anchor box on y-axis.
                h_gt (int): Height of the GT anchor.

            Returns:

            """
            overlap = max((h + h_gt) - (max((cy + h / 2), (cy_gt + h_gt / 2)) -
                                        min((cy - h / 2), (cy_gt - h_gt / 2))),
                          0)
            return overlap / (h + h_gt - overlap)

        data = {}
        # initialize data
        for y in range(self._height):
            for x in range(self._width):
                for z in range(self._k):
                    data[(y, x, z)] = (
                        -1,   # 1 text(iou > 0.7),
                              # 0 ignored(0.5 <= iou <= 0.7),
                              # -1 non-text(iou < 0.5)
                        0,    # -1 left side, 0 inside, 1 right side
                        0.0,  # vc
                        0.0,  # vh
                        0.0,  # offset
                        0.0   # iou
                    )

        # update data using anchors_list
        anchor_heights = self._get_anchor_heights()
        for anchors in anchors_list:
            for i, anchor in enumerate(anchors):
                idx_max = (0, 0, 0)
                result_max = None

                # variable with "_fm" means on feature maps, otherwise on image.
                x_gt, cy_gt, ah_gt = anchor
                x_fm = x_gt // self._fixed_width
                # todo x_fm out of range
                if x_fm >= self._width:
                    continue

                for y_fm in range(self._height):
                    for z, ah in enumerate(anchor_heights):
                        is_positive = False
                        # y_fm = torch.tensor([y_fm])
                        # ah = torch.tensor([ah])

                        cy = y_fm * self._fixed_width
                        iou = cal_iou(cy, ah, cy_gt, ah_gt)

                        o = 0
                        if i == 0:
                            side = -1
                            o = x_fm - x_gt / self._fixed_width - 1 / 2
                        elif i == len(anchors) - 1:
                            side = 1
                            o = x_fm - x_gt / self._fixed_width - 1 / 2
                        else:
                            side = 0

                        vc = 0.0
                        vh = 0.0
                        if iou < 0.5:
                            text = -1
                        elif iou > 0.7:
                            is_positive = True
                            text = 1
                            vc = (cy_gt - cy) / ah
                            vh = math.log(ah_gt / ah)
                        else:
                            text = 0

                        result = (text, side, vc, vh, o, iou)
                        if is_positive:
                            if data[(y_fm, int(x_fm), z)][0] != 1:
                                data[(y_fm, int(x_fm), z)] = result
                            else:
                                if data[(y_fm, int(x_fm), z)][-1] < iou:
                                    data[(y_fm, int(x_fm), z)] = result

                        if iou > 0.0:
                            if result_max is None or result[-1] > result_max[-1]:
                                idx_max = (y_fm, int(x_fm), z)
                                result_max = result

                # idx_max and iou_max is used to find the anchor
                # with highest iou overlap with a GT box if iou < 0.7.
                if result_max:
                    for yi in range(self._height):
                        for zi in range(self._k):
                            if data[(yi, int(x_fm), zi)][0] == 1:
                                break
                        else:
                            data[idx_max] = result_max

        return data

    def get(self, h: int, w: int, k: int):
        return self.data.get((h, w, k))


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
        self._use_cuda = True
        return super().cuda(device)

    def forward(self, x: torch.Tensor, targets: tuple):
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

        _, h, w, _ = x.shape
        index, anchors_list = targets
        anchor_data = AnchorData(int(index), anchors_list, h, w)
        choices = get_choices(h, w, self._k)

        n_pos = int(self._Ns * self._ratio)
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

        # print("n_pos = ", n_pos)
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
                loss_cls.append(self._Ls(x[:, yi, xi, start:end],
                                         torch.tensor([1]).cuda()))
                # vertical coordinates
                start = zi * 2
                end = start + 2
                loss_reg_v.append(self._Lv(x[:, yi, xi, start:end],
                                           torch.tensor([data[2:4]]).float().cuda()))
                # side-refinement
                if data[1] != 0:
                    start = self._k * 4 + zi
                    end = start + 1
                    loss_reg_o.append(self._Lo(x[:, yi, xi, start:end],
                                               torch.tensor([[data[-2]]]).float().cuda()))

            for yi, xi, zi in neg:
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[:, yi, xi, start:end],
                                         torch.tensor([0]).cuda()))
        else:
            for yi, xi, zi in pos:
                data = anchor_data.get(yi, xi, zi)
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[:, yi, xi, start:end],
                                         torch.tensor([1])))
                # vertical coordinates
                start = zi * 2
                end = start + 2
                loss_reg_v.append(self._Lv(x[:, yi, xi, start:end],
                                           torch.tensor([data[2:4]]).float()))
                # side-refinement
                if data[1] != 0:
                    start = self._k * 4 + zi
                    end = start + 1
                    loss_reg_o.append(self._Lo(x[:, yi, xi, start:end],
                                               torch.tensor([[data[-2]]]).float()))

            for yi, xi, zi in neg:
                # scores
                start = self._k * 2 + zi * 2
                end = start + 2
                loss_cls.append(self._Ls(x[:, yi, xi, start:end],
                                         torch.tensor([0])))

        loss = sum(loss_cls) / len(loss_cls)
        # todo loss_reg_v is supposed to be a non-empty list
        if len(loss_reg_v) > 0:
            loss += sum(loss_reg_v) / len(loss_reg_v)
        else:
            print("empty loss_reg_v")

        if len(loss_reg_o) > 0:
            loss += sum(loss_reg_o) / len(loss_reg_o)

        return loss
