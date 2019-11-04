import os
import cv2
from ...dataset import LabelDataset


class CRNNDataset(LabelDataset):

    def __getitem__(self, index: int):
        label = self.labels[index]
        key = self.keys[index]
        img_path = os.path.join(self._image_path, key)
        img = cv2.imread(img_path)
        if self._cvt_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._transformer is not None:
            img = self._transformer(img)

        return img, label

