import os
import re
import copy
import random
import cv2
from typing import Tuple
from torch.utils.data import Dataset
from deepsight.constants import IMG_EXTS


class SplitList(list):
    """List type for splitting in train_test_split()."""
    pass


class SplitDataset(Dataset):
    """Dataset that can be split into train dataset and test dataset using
     train_test_split(). For this purpose, the split attributes should be of
     type SplitList.
    """
    pass


def train_test_split(dataset: SplitDataset, test_size: float,
                     seed: int = 0) -> Tuple[SplitDataset, SplitDataset]:
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
        if isinstance(attr, SplitList):

            setattr(train_dataset, d, SplitList())
            setattr(test_dataset, d, SplitList())

            for i in idx_train:
                getattr(train_dataset, d).append(getattr(dataset, d)[i])

            for i in idx_test:
                getattr(test_dataset, d).append(getattr(dataset, d)[i])

    return train_dataset, test_dataset


class GroundTruthFolder(Dataset):
    """A image with ground truth data loader. The images and ground truth
    files should be arranged in this way:
    root/xxx.jpg
    root/xxx.jpg.gt
    ...
    The image will be ignored if the corresponding .gt file not exist or
    no coordinate data in the file.

    To support train_test_split(), the dataset attributes which will be split
    must be type of SplitList.

    Args:
        root (str): Root directory path.

    """

    def __init__(self, root: str):
        self._root = root
        self.img_paths = SplitList()
        self.gts = SplitList()
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


class LabelDataset(SplitDataset):

    def __init__(self, label_folder: str, image_folder: str):
        self._label_path = label_folder
        self._image_path = image_folder
        self.keys = SplitList()
        self.labels = SplitList()
        self._load_data()

    def __getitem__(self, index: int):
        key = self.keys[index]
        img_path = os.path.join(self._image_path, key)
        img = cv2.imread(img_path)
        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.keys)

    def _load_data(self):
        for f in os.listdir(self._label_path):
            if not f.lower().endswith(".txt"):
                continue
            path = os.path.join(self._label_path, f)
            with open(path) as label_file:
                for line in label_file:
                    try:
                        key, label = re.split("\s+", line.strip())
                        img_path = os.path.join(self._image_path, key)
                        if not os.path.exists(img_path):
                            continue
                        self.keys.append(key)
                        self.labels.append(label)
                    except:
                        continue
