import math
from ...predict import Predict
from .ctpn import ANCHOR_HEIGHTS
from ...utils.nms import nms

TEXT_NON_TEXT_THRESHOLD = 0.7
NMS_THRESH = 0.3


class CTPNPredict(Predict):

    def process(self, img):
        x = self._process(img)
        boxes = self._get_boxes(x, threshold=TEXT_NON_TEXT_THRESHOLD)
        if not boxes:
            print("no boxes")
            return []

        nms_boxes = [box[:4] for box in boxes]
        nms_score = [box[4].item() for box in boxes]
        idxes = nms(nms_boxes, nms_score, NMS_THRESH)
        boxes = [boxes[idx] for idx in idxes]
        groups = self._group_boxes(boxes)
        merged_boxes = self._merge_boxes(groups)

        return merged_boxes

    def _get_boxes(self, inputs, threshold: float, fixed_width: int = 16):
        boxes = []
        vcoords, scores, sides = inputs
        *_, height, width = scores.shape
        k = 10

        for y in range(height):
            for x in range(width):
                for z in range(k):
                    score = scores[0, 2 * z + 1, y, x]
                    if score > threshold:
                        vc = vcoords[0, 2 * z, y, x]
                        vh = vcoords[0, 2 * z + 1, y, x]
                        offset = sides[0, z, y, x] * fixed_width
                        ha = ANCHOR_HEIGHTS[z]

                        cy = vc * ha + y * fixed_width
                        h = math.exp(vh) * ha
                        left = math.floor(x * fixed_width - (fixed_width - 1) / 2)
                        right = math.ceil(x * fixed_width + (fixed_width - 1) / 2)
                        top = math.floor(cy - h / 2)
                        bottom = math.ceil(cy + h / 2)
                        boxes.append((left, top, right, bottom, score, offset))

        return boxes

    def _group_boxes(self, boxes):
        sorted_boxes = sorted(boxes, key=lambda x: x[0])   # sort according to left value
        groups = [[sorted_boxes[0]]]
        while len(sorted_boxes) > 0:
            box = sorted_boxes[0]
            left, top, right, bottom, *_ = box
            for group in groups:
                box1 = group[-1]
                left1, top1, right1, bottom1, _, o1 = box1
                if left1 - right >= 50:
                    continue
                if max(0, min(bottom, bottom1) - max(top, top1)) / \
                        (max(bottom, bottom1) - min(top, top1)) <= 0.7:
                    continue
                group.append(box)
                break
            else:
                groups.append([box])

            sorted_boxes = sorted_boxes[1:]

        return groups

    def _merge_boxes(self, sorted_groups, fixed_width: int = 16):
        merged = []
        for group in sorted_groups:
            offset_left = group[0][-1] * fixed_width
            offset_right = group[-1][-1] * fixed_width

            left = max(0, min([data[0] for data in group]) - offset_left)
            top = max(0, min([data[1] for data in group]))
            right = max([data[2] for data in group]) - offset_right
            bottom = max([data[3] for data in group])
            line_box = (left, top, right, bottom)

            merged.append(line_box)

        return merged
