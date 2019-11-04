import math
import cv2
from typing import Tuple, List
from deepsight.dataset import GroundTruthFolder, SplitList


def line_func(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = (y2 * x1 - y1 * x2) / (x1 - x2)
    return lambda x: a * x + b


class CTPNFolder(GroundTruthFolder):
    """CTPN dataset

    Args:
        root (str): Root directory path of the dataset.
        fixed_width (int): The fixed width of the text proposal.
        memorize (bool): Memorize to cache the anchors computation.
            Make sure there is enough memory, especially when you have
            a very large dataset.
    """
    _memory = {}

    def __init__(self, root: str, fixed_width: int = 16, short_side: int = 600,
                 memorize: bool = True, transformer=None):
        super().__init__(root)
        self.anchors = SplitList
        self._fixed_width = fixed_width
        self._short_side = short_side
        self._memorize = memorize
        self._transformer = transformer
        self._anchor_heights = self._get_anchor_heights()

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        if self._transformer is not None:
            img = self._transformer(img)

        if self._memorize:
            if index in self._memory:
                anchors_list = self._memory[index]
            else:
                scale = self._calc_scale(height, width)
                anchors_list = self._calc_anchors(index, scale)
                self._memory[index] = anchors_list
        else:
            scale = self._calc_scale(height, width)
            anchors_list = self._calc_anchors(index, scale)
            self._memory[index] = anchors_list

        return img, (index, anchors_list)

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
        k = 10
        s = -1
        return [round(self._fixed_width / r ** e) for e in range(s, k + s)]

    def _calc_scale(self, height: int, width: int) -> float:
        return self._short_side / min(height, width)

    def _convert_gts(self, index: int, scale: float) -> List[Tuple[float, ...]]:
        """
        Convert object ground truth according to the image scale.

        Notes:
            The result should not be converted to int, otherwise it will cause
            accuracy problem of slops of the gt box.
        """
        gts = []
        for x1, y1, x2, y2, x3, y3, x4, y4 in self.gts[index]:
            x1 = x1 * scale
            y1 = y1 * scale

            x2 = x2 * scale
            y2 = y2 * scale

            x3 = x3 * scale
            y3 = y3 * scale

            x4 = x4 * scale
            y4 = y4 * scale

            gts.append((x1, y1, x2, y2, x3, y3, x4, y4))

        return gts

    def _calc_anchors(self, index: int, scale: float) -> List[List[Tuple[int, float, float]]]:
        """
        Calculate anchors according to ground truth boxes.

        Args:
            gts (list): Text boxes with fixed width created from the original
                text ground truth.

        Returns:
            List[List[Tuple[pos (int), cy (float), h (float)]]]:
                pos: The left side position(x-axis) of the anchor box on
                    original image.
                cy: The center(y-axis) of the anchor box on the inputs image.
                h: The height of the anchor box on the inputs image.
        """
        anchors_list = []
        for gt_box in self._convert_gts(index, scale):
            x1, y1, x2, y2, x3, y3, x4, y4 = gt_box
            if (x1, y1) == (x2, y2) or (x2, y2) == (x3, y3) \
                    or (x3, y3) == (x4, y4) or (x4, y4) == (x1, y1):
                print("skip gt_box: ", gt_box)
                continue

            boxes = self._gt2anchors(gt_box)    # boxes of the same text line
            anchors = []    # anchors of the same text line
            for x1, y1, *_, y4 in boxes:
                cy = (y1 + y4) / 2
                h = float(y4 - y1)
                anchors.append((x1, cy, h))
            anchors_list.append(anchors)

        return anchors_list

    def _gt2anchors(self, gt_box: tuple):
        x01, y01, x02, y02, x03, y03, x04, y04 = gt_box
        xmin = min(x01, x02, x03, x04)
        xmax = max(x01, x02, x03, x04)
        n = math.ceil((xmax - xmin) / self._fixed_width)
        anchors = []
        for i in range(n):
            left = xmin + i * self._fixed_width
            right = left + self._fixed_width - 1
            y12, y34 = self._anchor_ys(gt_box, left, right)
            x1 = x4 = math.floor(left)
            x2 = x3 = x1 + self._fixed_width - 1
            y1 = y2 = math.floor(y12)
            y3 = y4 = math.ceil(y34)
            anchors.append((x1, y1, x2, y2, x3, y3, x4, y4))

        return anchors

    def _anchor_ys(self, gt_box: tuple, left: float, right: float):
        x1, y1, x2, y2, x3, y3, x4, y4 = gt_box

        if x1 <= x4:
            if x4 < x2:
                if left < x4:
                    if right <= x4:
                        y12 = line_func(x1, y1, x2, y2)(right)
                        y34 = line_func(x1, y1, x4, y4)(right)
                    elif right <= x2:
                        y12 = line_func(x1, y1, x2, y2)(right)
                        y34 = y4
                    else:
                        y12 = y2
                        y34 = y4
                elif left < x2:
                    if right <= x2:
                        y12 = line_func(x1, y1, x2, y2)(right)
                    else:
                        y12 = y2
                    y34 = line_func(x4, y4, x3, y3)(left)
                else:
                    y12 = line_func(x2, y2, x3, y3)(left)
                    y34 = line_func(x4, y4, x3, y3)(left)

            else:
                if left < x2:
                    if right <= x2:
                        y12 = line_func(x1, y1, x2, y2)(right)
                        y34 = line_func(x1, y1, x4, y4)(right)
                    elif right <= x4:
                        y12 = y2
                        y34 = line_func(x1, y1, x4, y4)(right)
                    else:
                        y12 = y2
                        y34 = y4
                elif left < x4:
                    y12 = line_func(x2, y2, x3, y3)(left)
                    if right <= x4:
                        y34 = line_func(x1, y1, x4, y4)(right)
                    else:
                        y34 = y4
                else:
                    y12 = line_func(x2, y2, x3, y3)(left)
                    y34 = line_func(x4, y4, x3, y3)(left)

        else:   # x1 > x4
            if x1 < x3:
                if left < x1:
                    if right <= x1:
                        y12 = line_func(x4, y4, x1, y1)(right)
                        y34 = line_func(x4, y4, x3, y3)(right)
                    elif right <= x3:
                        y12 = y1
                        y34 = line_func(x4, y4, x3, y3)(right)
                    else:
                        y12 = y1
                        y34 = y3
                elif left < x3:
                    y12 = line_func(x1, y1, x2, y2)(left)
                    if right <= x3:
                        y34 = line_func(x4, y4, x3, y3)(right)
                    else:
                        y34 = y3
                else:
                    y12 = line_func(x1, y1, x2, y2)(left)
                    y34 = line_func(x3, y3, x2, y2)(left)
            else:
                if left < x3:
                    if right <= x3:
                        y12 = line_func(x4, y4, x1, y1)(right)
                        y34 = line_func(x4, y4, x3, y3)(right)
                    elif right <= x1:
                        y12 = line_func(x4, y4, x1, y1)(right)
                        y34 = y3
                    else:
                        y12 = y1
                        y34 = y3
                elif left < x1:
                    if right <= x1:
                        y12 = line_func(x4, y4, x1, y1)(right)
                    else:
                        y12 = y1
                    y34 = line_func(x3, y3, x2, y2)(left)
                else:
                    y12 = line_func(x1, y1, x2, y2)(left)
                    y34 = line_func(x3, y3, x2, y2)(left)

        return y12, y34
