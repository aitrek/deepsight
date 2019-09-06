import os
import re
import copy
import random
import cv2
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import DatasetFolder, default_loader

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
                     seed: int=0) -> Tuple[GroundTruthFolder, GroundTruthFolder]:
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
