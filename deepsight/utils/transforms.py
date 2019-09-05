import numpy as np
from utils.image import resize


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

    def __call__(self, img: np.array) -> np.array:
        height, width, c = img.shape
        scale = min(self._height / height, self._width / width)
        height = int(height * scale)
        width = int(width * scale)
        img = resize(img, width=width, height=height)

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
