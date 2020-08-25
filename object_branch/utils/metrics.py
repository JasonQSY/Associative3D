"""Evaluation matric utils.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import pdb
import torch
from . import transformations

def volume_iou(pred, gt, thresh):
    gt = gt.float().ge(0.5)
    pred = pred.float().ge(thresh)
    intersection = torch.mul(gt, pred).sum()
    union = gt.sum() + pred.sum() - intersection
    return intersection/union

def quat_dist(pred, gt):
    try:
        rot_pred = transformations.quaternion_matrix(pred.numpy())
        rot_gt = transformations.quaternion_matrix(gt.numpy())
    except AttributeError:
        rot_pred = transformations.quaternion_matrix(pred)
        rot_gt = transformations.quaternion_matrix(gt)

    rot_rel = np.matmul(rot_pred, np.transpose(rot_gt))
    quat_rel = transformations.quaternion_from_matrix(rot_rel, isprecise=True)
    try:
        quat_rel = np.clip(quat_rel, a_min=-1, a_max=1)
        angle = math.acos(abs(quat_rel[0]))*360/math.pi
    except ValueError:
        pdb.set_trace()
    return angle

def direction_dist(pred, gt):
    dot = np.sum(pred*gt)
    angle = math.acos(dot)*180/math.pi
    return angle

def nms(dets, thresh, min_score=0):
    '''
    adapted from Fast R-CNN
    Copyright (c) 2015 Microsoft
    Licensed under The MIT License
    Written by Ross Girshick
    '''

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        if scores[i] < min_score:
            break

        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep