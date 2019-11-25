import math
import numpy as np
from ...predict import Predict
from .ctpn import ANCHOR_HEIGHTS
from ...utils.nms import nms

GROUP_DIST_THRESHOLD = 20
TEXT_NON_TEXT_THRESHOLD = 0.7
NMS_THRESH = 0.3
GROUP_MIN_SAMPLES = 3


class CTPNPredict(Predict):

    def process(self, img):
        x = self._process(img)
        boxes = self._get_boxes(x, threshold=TEXT_NON_TEXT_THRESHOLD)
        if not boxes:
            print("no boxes")
            return []

        nms_boxes = [box[:4] for box in boxes]
        nms_score = [box[4] for box in boxes]
        idxes = nms(nms_boxes, nms_score, NMS_THRESH)
        boxes = [boxes[idx] for idx in idxes]
        groups = self._group_boxes(boxes)
        merged_boxes = self._merge_boxes(groups)

        return merged_boxes, groups

    def _get_boxes(self, inputs, threshold: float, fixed_width: int = 16):
        boxes = []
        vcoords, scores, sides = inputs
        *_, height, width = scores.shape
        k = 10

        for y in range(height):
            for x in range(width):
                for z in range(k):
                    score = float(scores[0, 2 * z + 1, y, x])
                    if score > threshold:
                        vc = vcoords[0, 2 * z, y, x]
                        vh = vcoords[0, 2 * z + 1, y, x]
                        offset = float(sides[0, z, y, x] * fixed_width)
                        ha = ANCHOR_HEIGHTS[z]

                        cy = vc * ha + y * fixed_width
                        h = math.exp(vh) * ha
                        left = math.floor(x * fixed_width + (fixed_width - 1) / 2)
                        right = math.ceil((x + 1) * fixed_width + (fixed_width - 1) / 2)
                        top = math.floor(cy - h / 2)
                        bottom = math.ceil(cy + h / 2)
                        boxes.append((left, top, right, bottom, score, offset))

        return boxes

    def _group_boxes(self, boxes):
        boxes = sorted(boxes, key=lambda x: x[0])   # sort according to left value
        groups = [[boxes[0]]]
        while len(boxes) > 0:
            box = boxes[0]
            left, top, right, bottom, *_ = box
            for group in groups:
                g_left, g_top, g_right, g_bottom, *_ = group[-1]
                if left - g_right >= GROUP_DIST_THRESHOLD:
                    continue
                if max(0, min(bottom, g_bottom) - max(top, g_top)) / \
                        (max(bottom, g_bottom) - min(top, g_top)) <= 0.7:
                    continue
                group.append(box)
                break
            else:
                groups.append([box])

            boxes = boxes[1:]

        groups = [group for group in groups if len(group) > GROUP_MIN_SAMPLES]

        return groups

    def _merge_boxes(self, sorted_groups, fixed_width: int = 16):
        # todo offset
        merged = []
        for group in sorted_groups:
            # left, top, right, bottom, score, offset = group
            group = np.array(group)
            lefts = group[:, 0]
            tops = group[:, 1]
            rights = group[:, 2]
            bottoms = group[:, 3]

            xmin = min(lefts)
            xmax = max(rights)

            center_xs = lefts + (fixed_width - 1) / 2
            center_ys = (tops + bottoms) / 2

            poly_center_params = np.polyfit(center_xs, center_ys, 1)    # [k, b]
            poly_center = np.poly1d(poly_center_params)

            center_ys_hat = poly_center(center_xs)
            top_gaps = center_ys_hat - tops
            bottom_gaps = bottoms - center_ys_hat

            avg_top_gap = np.mean(top_gaps)
            avg_bottom_gap = np.mean(bottom_gaps)

            poly_top_params = [poly_center_params[0],
                               poly_center_params[1] + avg_top_gap]
            poly_bottom_params = [poly_center_params[0],
                                  poly_center_params[1] - avg_bottom_gap]

            poly_top = np.poly1d(poly_top_params)
            poly_bottom = np.poly1d(poly_bottom_params)

            x1 = x4 = xmin
            x2 = x3 = xmax
            y1 = poly_top(x1)
            y2 = poly_top(x2)
            y3 = poly_bottom(x3)
            y4 = poly_bottom(x4)

            merged.append((x1, y1, x2, y2, x3, y3, x4, y4))

        return merged
