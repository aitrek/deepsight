import os
import re
import copy
import math
import random
import cv2
from typing import Tuple
from torch.utils.data import Dataset

IMG_EXTS = [".jpe", ".jpg", ".jpeg", ".png"]


class GroundTruthFolder(Dataset):
    """A image with ground truth data loader. The images and ground truth
    files should be arranged in this way:
    root/xxx.jpg
    root/xxx.jpg.gt
    ...

    Args:
        root (str): Root directory path.

    """

    def __init__(self, root: str):
        self._root = root
        self.img_paths = []
        self.gts = []
        self._load_data(self._root)

    def __getitem__(self, index: int):
        img = cv2.imread(self.img_paths[index])
        gt = self.gts[index]
        return img, gt

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
                    self.img_paths.append(path)
                    self.gts.append(self._get_gt(path + ".gt"))

    @staticmethod
    def _get_gt(gt_path: str):
        gt = []
        try:
            with open(gt_path) as gt_file:
                for line in gt_file:
                    gt.append((int(t) for t in re.split("\s+", line.strip())))
        except FileNotFoundError:
            return []


def train_test_split(dataset: GroundTruthFolder,
                     test_size: float,
                     seed: int = 0
                     ) -> Tuple[GroundTruthFolder, GroundTruthFolder]:
    """
    Random split ground truth dataset into train and test subsets.

    Args:
        dataset (GroundTruthFolder): The dataset to be split.
        test_size (float): The proportion of the dataset to include in
            the test split.
        seed (int): pseudo-random number

    Returns:
        Tuple of train dataset and test dataset.
    """
    train = copy.copy(dataset)
    train.img_paths = []
    train.gts = []
    test = copy.copy(dataset)
    test.img_paths = []
    test.gts = []

    l_all = len(dataset)
    l_test = int(l_all * test_size)
    l_train = l_all - l_test

    random.seed(seed)
    idx = random.sample(range(l_all), l_all)
    idx_train = idx[:l_train]
    idx_test = idx[l_train:]

    for i in idx_train:
        train.img_paths.append(dataset.img_paths[i])
        train.gts.append(dataset.gts[i])

    for i in idx_test:
        test.img_paths.append(dataset.img_paths[i])
        test.gts.append(dataset.gts[i])

    return train, test


class TextProposalFolder(GroundTruthFolder):

    def __init__(self, data_folder: str, fixed_width: int = 16):
        super().__init__(data_folder)
        self._fixed_width = fixed_width
        self._convert_gts()

    def _convert_gts(self):
        """
        Convert object ground truth to sequential fixed-width
        fine-scale text proposals(anchors).
        """
        anchors = []
        for gt in self.gts:
            anch = []
            for t in gt:
                anch += self._gt2anchors(t)
            anchors.append(anch)

        self.gts = anchors

    def _gt2anchors(self, gt_pts, w=16):
        x01, y01, x02, y02, x03, y03, x04, y04 = gt_pts
        xmin = min(x01, x04)
        n = math.ceil((max(x02, x03) - min(x01, x04)) / w)
        anchors = []
        for i in range(n):
            x1 = x4 = math.floor(xmin + i * w)
            x2 = x3 = x1 + w
            y1, y4 = self._anchor_ys(gt_pts, x1)
            y2, y3 = self._anchor_ys(gt_pts, x2)
            y1 = y2 = math.floor(min(y1, y2))
            y3 = y4 = math.ceil(max(y3, y4))
            anchors.append((x1, y1, x2, y2, x3, y3, x4, y4))

        return anchors

    @staticmethod
    def _line_fn(x1, y1, x2, y2):
        a = (y1 - y2) / (x1 - x2)
        b = (y2 * x1 - y1 * x2) / (x1 - x2)
        return lambda x: a * x + b

    def _anchor_ys(self, gt_pts, x, w=16):
        x1, y1, x2, y2, x3, y3, x4, y4 = gt_pts
        if x > max(x2, x3):
            x -= w
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
