import os
import re
import copy
import random
import cv2
from typing import Tuple
from torch.utils.data import Dataset
from ..constants import IMG_EXTS


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
                    boxes.append([int(t) for t in re.split("\s+", line.strip())])
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
