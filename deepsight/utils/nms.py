"""Non-Maximum Suppression"""
import numpy as np
from typing import Sequence, Tuple, List

BoxType = Tuple[int, int, int, int]     # [left, top, right, bottom]


def nms(boxes: Sequence[BoxType], scores: Sequence[float], threshold: float) -> List[int]:
    """

    Args:
        boxes: List of box coordinates in form of [left, top, right, bottom].
        scores: Confidence scores of boxes to be an object.
        threshold: Threshold of overlap.

    Returns:
        Indexes of picked boxes.
    """
    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    lefts = boxes[:, 0]
    tops = boxes[:, 1]
    rights = boxes[:, 2]
    bottoms = boxes[:, 3]

    areas = (rights - lefts + 1) * (bottoms - tops + 1)
    order = np.argsort(scores)

    picks = []
    while order.size > 0:
        idx = order[-1]
        picks.append(idx)

        overlap_lefts = np.maximum(lefts[idx], lefts[order[:-1]])
        overlap_tops = np.maximum(tops[idx], tops[order[:-1]])
        overlap_rights = np.minimum(rights[idx], rights[order[:-1]])
        overlap_bottoms = np.minimum(bottoms[idx], bottoms[order[:-1]])

        ws = np.maximum(0.0, overlap_rights - overlap_lefts + 1)
        hs = np.maximum(0.0, overlap_bottoms - overlap_tops + 1)

        overlap_areas = ws * hs

        ratios = overlap_areas / (areas[idx] + areas[order[:-1]] - overlap_areas)

        order = order[np.where(ratios < threshold)]

    return picks
