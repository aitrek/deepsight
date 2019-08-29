"""Utilities for ground truth"""
import math


def line_fn(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = (y2 * x1 - y1 * x2) / (x1 - x2)
    return lambda x: a * x + b


def anchor_ys(gt_pts, x, w=16):
    x1, y1, x2, y2, x3, y3, x4, y4 = gt_pts
    if x > max(x2, x3):
        x -= w
    if x1 < x4:
        if x4 < x2:
            if x <= x4:
                y12 = line_fn(x1, y1, x2, y2)(x)
                y34 = line_fn(x1, y1, x4, y4)(x)
            elif x4 < x <= x2:
                y12 = line_fn(x1, y1, x2, y2)(x)
                y34 = line_fn(x4, y4, x3, y3)(x)
            else:
                y12 = line_fn(x2, y2, x3, y3)(x)
                y34 = line_fn(x4, y4, x3, y3)(x)
        else:
            if x <= x2:
                y12 = line_fn(x1, y1, x2, y2)(x)
                y34 = line_fn(x1, y1, x4, y4)(x)
            elif x2 < x <= x4:
                y12 = line_fn(x2, y2, x3, y3)(x)
                y34 = line_fn(x1, y1, x4, y4)(x)
            else:
                y12 = line_fn(x2, y2, x3, y3)(x)
                y34 = line_fn(x4, y4, x3, y3)(x)

    elif x1 > x4:
        if x1 < x3:
            if x <= x1:
                y12 = line_fn(x4, y4, x1, y1)(x)
                y34 = line_fn(x4, y4, x3, y3)(x)
            elif x1 < x <= x3:
                y12 = line_fn(x1, y1, x2, y2)(x)
                y34 = line_fn(x4, y4, x3, y3)(x)
            else:
                y12 = line_fn(x1, y1, x2, y2)(x)
                y34 = line_fn(x3, y3, x2, y2)(x)
        else:
            if x <= x3:
                y12 = line_fn(x4, y4, x1, y1)(x)
                y34 = line_fn(x4, y4, x3, y3)(x)
            elif x3 < x <= x1:
                y12 = line_fn(x4, y4, x1, y1)(x)
                y34 = line_fn(x3, y3, x2, y2)(x)
            else:
                y12 = line_fn(x1, y1, x2, y2)(x)
                y34 = line_fn(x3, y3, x2, y2)(x)
    else:
        y12 = y1
        y34 = y4

    return y12, y34


def gt2anchors(gt_pts, w=16):
    x01, y01, x02, y02, x03, y03, x04, y04 = gt_pts
    xmin = min(x01, x04)
    n = math.ceil((max(x02, x03) - min(x01, x04)) / w)
    anchors = []
    for i in range(n):
        x1 = x4 = math.floor(xmin + i * w)
        x2 = x3 = x1 + w
        y1, y4 = anchor_ys(gt_pts, x1)
        y2, y3 = anchor_ys(gt_pts, x2)
        y1 = y2 = math.floor(min(y1, y2))
        y3 = y4 = math.ceil(max(y3, y4))
        anchors.append((x1, y1, x2, y2, x3, y3, x4, y4))

    return anchors
