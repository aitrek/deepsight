import cv2
import numpy as np


class BGR2GRAY:

    def __call__(self, img: np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class SameHeightTransform:

    def __init__(self, height: int):
        self._height = height

    def __call__(self, img: np.ndarray):
        h = img.shape[0]
        w = img.shape[1]
        scale = self._height / h
        resized_height = int(h * scale)
        resized_width = int(w * scale)
        resized_img = cv2.resize(img, (resized_width, resized_height),
                                 cv2.INTER_NEAREST)
        return resized_img
