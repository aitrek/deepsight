import cv2
import numpy as np


def resize(img: np.array, width: int = None, height: int = None) -> np.array:
    """

    Args:
        img:
        width:
        height:

    Returns:

    """
    shape = img.shape[:2]
    if width is None or height is None:
        if width is None and height is None:
            height, width = shape
        else:
            if width is None:
                width = int(height * shape[1] / shape[0])
            else:
                height = int(width * shape[0] / shape[1])

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
