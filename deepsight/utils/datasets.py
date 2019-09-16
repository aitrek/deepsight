import os
import re
import copy
import math
import random
import cv2
from typing import Tuple, List
from torch.utils.data import Dataset
from functools import lru_cache

IMG_EXTS = [".jpe", ".jpg", ".jpeg", ".png"]


class DataList(list):
    pass


class GroundTruthFolder(Dataset):
    """A image with ground truth data loader. The images and ground truth
    files should be arranged in this way:
    root/xxx.jpg
    root/xxx.jpg.gt
    ...
    The image will be ignored if the corresponding .gt file not exist or
    no coordinate data in the file.

    To support train_test_split(), the dataset attributes which will be split
    must be type of DataList.

    Args:
        root (str): Root directory path.

    """

    def __init__(self, root: str):
        self._root = root
        self.img_paths = DataList()
        self.gts = DataList()
        self._load_data(self._root)

    def __getitem__(self, index: int):
        img = cv2.imread(self.img_paths[index])
        boxes = self.gts[index]
        return img, boxes

    def __len__(self):
        return len(self.img_paths)

    def _load_data(self, root: str):
        root = os.path.abspath(root)
        for f in os.listdir(root):
            path = os.path.join(root, f)
            if os.path.isdir(path):
                self._load_data(path)
            else:
                if f.lower().split(".")[-1] in IMG_EXTS:
                    boxes = self._get_boxes(path + ".gt")
                    if boxes:
                        self.img_paths.append(path)
                        self.gts.append(boxes)

    @staticmethod
    def _get_boxes(gt_path: str):
        boxes = []
        try:
            with open(gt_path) as gt_file:
                for line in gt_file:
                    boxes.append((int(t) for t in re.split("\s+", line.strip())))
        except FileNotFoundError:
            pass
        return boxes


def train_test_split(dataset: GroundTruthFolder, test_size: float,
                     seed: int = 0) -> Tuple[GroundTruthFolder, GroundTruthFolder]:
    """
    Random split ground truth dataset into train_dataset and test_dataset subsets.

    Args:
        dataset (GroundTruthFolder): The dataset to be split.
        test_size (float): The proportion of the dataset to include in
            the test_dataset split.
        seed (int): pseudo-random number

    Returns:
        Tuple of train_dataset dataset and test_dataset dataset.
    """
    train_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)

    l_all = len(dataset)
    l_test = int(l_all * test_size)
    l_train = l_all - l_test

    random.seed(seed)
    idx = random.sample(range(l_all), l_all)
    idx_train = idx[:l_train]
    idx_test = idx[l_train:]

    for d in dir(dataset):
        attr = getattr(dataset, d)
        if isinstance(attr, DataList):

            setattr(train_dataset, d, DataList())
            setattr(test_dataset, d, DataList())

            for i in idx_train:
                getattr(train_dataset, d).append(getattr(dataset, d)[i])

            for i in idx_test:
                getattr(test_dataset, d).append(getattr(dataset, d)[i])

    return train_dataset, test_dataset


class CTPNFolder(GroundTruthFolder):
    """CTPN dataset

    Args:
        root (str): Root directory path of the dataset.
        fixed_width (int): The fixed width of the text proposal.
        memorize (bool): Memorize to cache the anchors computation.
            Make sure there is enough memory, especially when you have
            a very large dataset.
    """

    def __init__(self, root: str, fixed_width: int = 16, short_side: int = 600,
                 memorize: bool = True):
        super().__init__(root)
        self.anchors = DataList
        self._fixed_width = fixed_width
        self._short_side = short_side
        self._memorize = memorize
        self._anchor_heights = self._get_anchor_heights()

    def __getitem__(self, index: int):
        img = cv2.imread(self.img_paths[index])
        height, width, _ = img.shape
        scale = self._calc_scale(height, width)
        resized_height = int(height * scale)
        resized_width = int(width * scale)

        resized_img = cv2.resize(img,
                                 (resized_width, resized_height),
                                 interpolation=cv2.INTER_NEAREST)
        resized_gts = self._convert_gts(index, scale)
        anchors = self._mem_calc_anchors(resized_gts) if self._memorize \
            else self._calc_anchors(resized_gts)

        return resized_img, anchors

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
        return min(height, width) / self._short_side

    def _convert_gts(self, index: int, scale: float) -> List[Tuple]:
        """
        Convert object ground truth according to the image scale.
        """
        gts = []
        for x1, y1, x2, y2, x3, y3, x4, y4 in self.gts[index]:
            x1 = math.floor(x1 * scale)
            y1 = math.floor(y1 * scale)

            x2 = math.ceil(x2 * scale)
            y2 = math.floor(y2 * scale)

            x3 = math.ceil(x3 * scale)
            y3 = math.ceil(y3 * scale)

            x4 = math.floor(x4 * scale)
            y4 = math.ceil(y4 * scale)

            gts.append((x1, y1, x2, y2, x3, y3, x4, y4))

        return gts

    @lru_cache(maxsize=None)
    def _mem_calc_anchors(self, gts: list) -> List[List[Tuple[int, float, int]]]:
        return self._calc_anchors(gts)

    def _calc_anchors(self, gts: list) -> List[List[Tuple[int, float, int]]]:
        """
        Calculate anchors according to ground truth boxes.

        Args:
            boxes (list): All the text boxes with fixed width created from
                the original text ground truth.


        Returns:
            List[List[Tuple[pos (int), cy (float), h (int)]]]:
                pos: The left side position(x-axis) of the anchor box on
                    original image.
                cy: The center(y-axis) of the anchor box on the input image.
                h: The height of the anchor box on the input image.
        """
        anchors_list = []
        for gt_box in gts:
            boxes = self._gt2anchors(gt_box)    # boxes of the same text line
            anchors = []    # anchors of the same text line
            for x1, y1, *_, y4 in boxes:
                cy = (y1 + y4) / 2
                h = y4 - y1
                anchors.append((x1, cy, h))
            anchors_list.append(anchors)

        return anchors_list

    def _gt2anchors(self, gt_box: tuple):
        x01, y01, x02, y02, x03, y03, x04, y04 = gt_box
        xmin = min(x01, x04)
        n = math.ceil((max(x02, x03) - min(x01, x04)) / self._fixed_width)
        anchors = []
        for i in range(n):
            x1 = x4 = math.floor(xmin + i * self._fixed_width)
            x2 = x3 = x1 + self._fixed_width
            y1, y4 = self._anchor_ys(gt_box, x1)
            y2, y3 = self._anchor_ys(gt_box, x2)
            y1 = y2 = math.floor(min(y1, y2))
            y3 = y4 = math.ceil(max(y3, y4))
            anchors.append((x1, y1, x2, y2, x3, y3, x4, y4))

        return anchors

    @staticmethod
    def _line_fn(x1, y1, x2, y2):
        a = (y1 - y2) / (x1 - x2)
        b = (y2 * x1 - y1 * x2) / (x1 - x2)
        return lambda x: a * x + b

    def _anchor_ys(self, gt_box: tuple, x):
        x1, y1, x2, y2, x3, y3, x4, y4 = gt_box
        if x > max(x2, x3):
            x -= self._fixed_width
        if x1 < x4:
            if x4 < x2:
                if x <= x4:
                    y12 = self._line_fn(x1, y1, x2, y2)(x)
                    y34 = self._line_fn(x1, y1, x4, y4)(x)
                elif x4 < x <= x2:
                    y12 = self._line_fn(x1, y1, x2, y2)(x)
                    y34 = self._line_fn(x4, y4, x3, y3)(x)
                else:
                    y12 = self._line_fn(x2, y2, x3, y3)(x)
                    y34 = self._line_fn(x4, y4, x3, y3)(x)
            else:
                if x <= x2:
                    y12 = self._line_fn(x1, y1, x2, y2)(x)
                    y34 = self._line_fn(x1, y1, x4, y4)(x)
                elif x2 < x <= x4:
                    y12 = self._line_fn(x2, y2, x3, y3)(x)
                    y34 = self._line_fn(x1, y1, x4, y4)(x)
                else:
                    y12 = self._line_fn(x2, y2, x3, y3)(x)
                    y34 = self._line_fn(x4, y4, x3, y3)(x)

        elif x1 > x4:
            if x1 < x3:
                if x <= x1:
                    y12 = self._line_fn(x4, y4, x1, y1)(x)
                    y34 = self._line_fn(x4, y4, x3, y3)(x)
                elif x1 < x <= x3:
                    y12 = self._line_fn(x1, y1, x2, y2)(x)
                    y34 = self._line_fn(x4, y4, x3, y3)(x)
                else:
                    y12 = self._line_fn(x1, y1, x2, y2)(x)
                    y34 = self._line_fn(x3, y3, x2, y2)(x)
            else:
                if x <= x3:
                    y12 = self._line_fn(x4, y4, x1, y1)(x)
                    y34 = self._line_fn(x4, y4, x3, y3)(x)
                elif x3 < x <= x1:
                    y12 = self._line_fn(x4, y4, x1, y1)(x)
                    y34 = self._line_fn(x3, y3, x2, y2)(x)
                else:
                    y12 = self._line_fn(x1, y1, x2, y2)(x)
                    y34 = self._line_fn(x3, y3, x2, y2)(x)
        else:
            y12 = y1
            y34 = y4

        return y12, y34
