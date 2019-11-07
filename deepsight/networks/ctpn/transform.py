import cv2
import numpy as np


class UnifySize:
    """
    Unify image with same width or height, and the marginal areas will be
    filled with 0.

    Args:
        width (int): The image width after unifying.
        height (int): The image height after unifying
        align (str): The alignment to put the original image.
            Default:
                "lt" - left top
            Alternatives:
                "c" - center
    """
    def __init__(self, width: int, height: int, align: str = "lt"):
        self._width = width
        self._height = height
        self._align = align

    def __call__(self, img: np.ndarray) -> np.ndarray:
        height, width, c = img.shape
        scale = min(self._height / height, self._width / width)
        height = int(height * scale)
        width = int(width * scale)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

        target = np.zeros((self._height, self._width, c), dtype=np.uint8)
        if self._align == "c":
            if height == self._height:
                left = int((self._width - width) / 2)
                right = left + width
                target[:, left:right, :] = img
            else:
                top = int((self._height - height) / 2)
                bottom = top + height
                target[top:bottom, :, :] = img
        else:
            target[0:height, 0:width] = img

        return target


class ShortSideTransform:
    """
    Resize image by setting its short side to 600 with original aspect ratio.

    Args:
        short_side (int): The target image's short side.
    """
    def __init__(self, short_side: int):
        self._short_side = short_side

    def __call__(self, img: np.ndarray):
        height, width, _ = img.shape
        scale = self._short_side / min(height, width)
        resized_height = int(height * scale)
        resized_width = int(width * scale)
        resized_img = cv2.resize(img, (resized_width, resized_height),
                                 cv2.INTER_NEAREST)
        return resized_img

    def __repr__(self):
        return self.__class__.__name__ + '(short_side={0})'.\
            format(self._short_side)
