import math
import random
import torch
import torch.nn as nn
from typing import List, Tuple
from .ctpn import ANCHOR_HEIGHTS


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
        for anchors in anchors_list:
            for i, anchor in enumerate(anchors):
                idx_max = (0, 0, 0)
                result_max = None
                need_max = True

                # variable with "_fm" means on feature maps, otherwise on image.
                x_gt, cy_gt, ah_gt = anchor
                x_fm = x_gt // self._fixed_width
                if x_fm < 0 or x_fm >= self._width:
                    continue

                for y_fm in range(self._height):
                    for z, ah in enumerate(ANCHOR_HEIGHTS):
                        cy_a = y_fm * self._fixed_width
                        iou = cal_iou(cy_a, ah, cy_gt, ah_gt)

                        # the side-refinement offset is changed to be the gap
                        # between left/right side to the x_min of left/rightmost
                        # anchor's x_min
                        if i == 0:  # leftmost
                            side = -1
                            o = x_gt / self._fixed_width - x_fm
                        elif i == len(anchors) - 1:  # rightmost
                            side = 1
                            o = x_gt / self._fixed_width - x_fm
                        else:
                            side = 0
                            o = 0.0

                        text = -1
                        vc = (cy_gt - cy_a) / ah
                        vh = math.log(ah_gt / ah)
                        result = (text, side, vc, vh, o, iou)

                        if iou > 0.7:
                            need_max = False
                            text = 1
                            result = (text, side, vc, vh, o, iou)
                            if data[(y_fm, int(x_fm), z)][0] != 1:
                                data[(y_fm, int(x_fm), z)] = result
                            else:
                                if data[(y_fm, int(x_fm), z)][-1] < iou:
                                    data[(y_fm, int(x_fm), z)] = result

                        if need_max and iou > 0.0:
                            if result_max is None or iou > result_max[-1]:
                                idx_max = (y_fm, int(x_fm), z)
                                result_max = (1, *result[1:])

                # idx_max and iou_max is used to find the anchor
                # with highest iou overlap with a GT box if iou < 0.7.
                if need_max:
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

    def forward(self, input: tuple, targets: tuple):
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

        vcoords, scores, sides = input
        # _, h, w, _ = x.shape
        _, _, h, w = scores.shape
        index, anchors_list = targets
        anchor_data = AnchorData(int(index), anchors_list, h, w)
        choices = get_choices(h, w, self._k)

        n_pos = int(self._Ns * self._ratio)
        n_neg = self._Ns - n_pos

        pos = []
        neg = []
        for yi, xi, zi in choices:
            data = anchor_data.get(yi, xi, zi)
            if data[0] == 1 and len(pos) < self._Ns:
                pos.append((yi, xi, zi))
            elif data[0] == -1 and len(neg) < self._Ns:
                neg.append((yi, xi, zi))
            else:
                continue

            if len(pos) > n_pos and len(neg) > n_neg:
                break

        # print("n_pos = ", n_pos)
        pos = pos[:n_pos]
        neg = neg[:self._Ns-len(pos)]   # the length of neg is supposed much larger than self._Ns

        loss_cls = []
        loss_reg_v = []
        loss_reg_o = []

        if self._use_cuda:
            for yi, xi, zi in pos:
                data = anchor_data.get(yi, xi, zi)
                # scores
                loss_cls.append(self._Ls(scores[:, zi * 2:(zi + 1) * 2, yi, xi],
                                         torch.tensor([1]).cuda()))
                # vertical coordinates
                loss_reg_v.append(self._Lv(vcoords[:, zi * 2:(zi + 1) * 2, yi, xi],
                                           torch.tensor([data[2:4]]).float().cuda()))
                # side-refinement
                if data[1] != 0:
                    loss_reg_o.append(self._Lo(sides[:, zi:(zi + 1), yi, xi],
                                               torch.tensor([[data[-2]]]).float().cuda()))

            for yi, xi, zi in neg:
                # scores
                loss_cls.append(self._Ls(scores[:, zi * 2:(zi + 1) * 2, yi, xi],
                                         torch.tensor([0]).cuda()))
        else:
            for yi, xi, zi in pos:
                data = anchor_data.get(yi, xi, zi)
                # scores
                loss_cls.append(self._Ls(scores[:, zi * 2:(zi + 1) * 2, yi, xi],
                                         torch.tensor([1])))
                # vertical coordinates
                loss_reg_v.append(self._Lv(vcoords[:, zi * 2:(zi + 1) * 2, yi, xi],
                                           torch.tensor([data[2:4]]).float()))
                # side-refinement
                if data[1] != 0:
                    loss_reg_o.append(self._Lo(sides[:, zi:(zi + 1), yi, xi],
                                               torch.tensor([[data[-2]]]).float()))

            for yi, xi, zi in neg:
                # scores
                loss_cls.append(self._Ls(scores[:, zi * 2:(zi + 1) * 2, yi, xi],
                                         torch.tensor([0])))

        loss = 0.0
        avg_loss_cls = sum(loss_cls) / len(loss_cls)
        loss += avg_loss_cls

        avg_loss_reg_v = 0.0
        if len(loss_reg_v) > 0:
            avg_loss_reg_v = sum(loss_reg_v) / len(loss_reg_v)
            loss += self._lambda1 * avg_loss_reg_v

        avg_loss_reg_o = 0.0
        if len(loss_reg_o) > 0:
            avg_loss_reg_o = sum(loss_reg_o) / len(loss_reg_o)
            loss += self._lambda2 * avg_loss_reg_o

        return loss, avg_loss_cls, avg_loss_reg_v, avg_loss_reg_o
