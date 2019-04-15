
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from bbox import bbox_iou
# from post_process_abcd_stop import BBox


S_CLASS, T_CLASS, O_CLASS, P_CLASS, OCTAGON_CLASS = map(float, range(4,9))
S_CLASS_INT, T_CLASS_INT, O_CLASS_INT, P_CLASS_INT, OCTAGON_CLASS_INT = range(4,9)


def compute_score(bound_boxes, real_bboxes_repeated, class_int, pred_repeated, pred_unsqueezed, num_of_boxes):
    bound_boxes_unsqueezed = bound_boxes.unsqueeze(2)

    t_cmp = (bound_boxes_unsqueezed - real_bboxes_repeated)[...,:4] * torch.tensor([[1.,1.,-1.,-1.]], device=pred_repeated.device)
    # remove pred_unsqueeze itself
    # inside_prediction = pred_repeated * (t_cmp != 0.).all(dim=3, keepdim=True).float()

    delta_bound = torch.tensor([[0., -5., 0., -5.]], device=pred_repeated.device)
    inside_prediction = pred_repeated * (t_cmp < delta_bound).all(dim=3, keepdim=True).float()

    # only choose the class with max confidence for the bbox
    inside_prediction = inside_prediction * (inside_prediction[..., 5:].max(dim=3, keepdim=True)[1] == class_int).float()

    class_score = (inside_prediction[..., 5+class_int] * inside_prediction[...,4]).max(dim=2)[0]

    return class_score


def post_predict_transform(prediction, grid_xy, anchor_wh, stride, nC, bs, CUDA = True):
    const_noise = torch.exp(torch.tensor([-16.], device=prediction.device))
    # same from predict_transform, if needed,
    # but s/t/o/p/octagon may be detected by different yolo layers
    # so we post process two concatenated results of the two yolo layers
    print('prediction.shape', prediction.shape)

    # vectorize the for loop
    prediction_clone = prediction.clone()

    # calculate the bboxes from prediction
    prediction_clone[..., 0:2] = torch.sigmoid(prediction[..., 0:2]) + grid_xy  # xy
    prediction_clone[..., 2:4] = torch.exp(prediction[..., 2:4]) * anchor_wh  # wh yolo method
    prediction_clone[..., 4] = torch.sigmoid(prediction[..., 4])  # p_conf
    prediction_clone[..., :4] *= stride
    prediction_clone[..., 5:] = torch.sigmoid(prediction[..., 5:])  # p_class

    # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
    prediction_clone = prediction_clone.view(bs, -1, 5 + nC)
    print('prediction_clone.shape', prediction_clone.shape)
    num_of_boxes = prediction_clone.size(1)

    real_bboxes = prediction_clone.new(prediction_clone.size(0), num_of_boxes, 4)
    print('real_bboxes.shape', real_bboxes.shape)
    real_bboxes[:,:,0] = (prediction_clone[:,:,0] - prediction_clone[:,:,2]/2)
    real_bboxes[:,:,1] = (prediction_clone[:,:,1] - prediction_clone[:,:,3]/2)
    real_bboxes[:,:,2] = (prediction_clone[:,:,0] + prediction_clone[:,:,2]/2)
    real_bboxes[:,:,3] = (prediction_clone[:,:,1] + prediction_clone[:,:,3]/2)

    real_bboxes_repeated = real_bboxes.unsqueeze(1).repeat(1, num_of_boxes, 1, 1)

    # compute the boxes left top right bottom
    # This boxes are for S
    boxes_left = prediction_clone.new(real_bboxes.shape)
    boxes_left[:,:,0] = (prediction_clone[:,:,0] - prediction_clone[:,:,2]/2)
    boxes_left[:,:,1] = (prediction_clone[:,:,1] - prediction_clone[:,:,3]/2)
    boxes_left[:,:,2] = (prediction_clone[:,:,0] - prediction_clone[:,:,2]*0.1)
    boxes_left[:,:,3] = (prediction_clone[:,:,1] + prediction_clone[:,:,3]/2)

    boxes_left_middle = prediction_clone.new(real_bboxes.shape)
    boxes_left_middle[:,:,0] = (prediction_clone[:,:,0] - prediction_clone[:,:,2]/4)
    boxes_left_middle[:,:,1] = (prediction_clone[:,:,1] - prediction_clone[:,:,3]/2)
    boxes_left_middle[:,:,2] = prediction_clone[:,:,0] + prediction_clone[:,:,2]*0.1
    boxes_left_middle[:,:,3] = (prediction_clone[:,:,1] + prediction_clone[:,:,3]/2)

    boxes_middle_right = prediction_clone.new(real_bboxes.shape)
    boxes_middle_right[:,:,0] = prediction_clone[:,:,0] - prediction_clone[:,:,2]*0.1
    boxes_middle_right[:,:,1] = (prediction_clone[:,:,1] - prediction_clone[:,:,3]/2)
    boxes_middle_right[:,:,2] = (prediction_clone[:,:,0] + prediction_clone[:,:,2]/4)
    boxes_middle_right[:,:,3] = (prediction_clone[:,:,1] + prediction_clone[:,:,3]/2)

    boxes_right = prediction_clone.new(real_bboxes.shape)
    boxes_right[:,:,0] = (prediction_clone[:,:,0] + prediction_clone[:,:,2]*0.1)
    boxes_right[:,:,1] = (prediction_clone[:,:,1] - prediction_clone[:,:,3]/2)
    boxes_right[:,:,2] = (prediction_clone[:,:,0] + prediction_clone[:,:,2]/2)
    boxes_right[:,:,3] = (prediction_clone[:,:,1] + prediction_clone[:,:,3]/2)

    pred_repeated = prediction_clone.unsqueeze(1).repeat(1, num_of_boxes, 1, 1)
    pred_unsqueezed = prediction_clone.unsqueeze(2)

    # consider iff. S/T/O/P appears inside the specific area of the octagon. has order
    s_class_score = compute_score(boxes_left, real_bboxes_repeated, S_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)
    t_class_score = compute_score(boxes_left_middle, real_bboxes_repeated, T_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)
    o_class_score = compute_score(boxes_middle_right, real_bboxes_repeated, O_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)
    p_class_score = compute_score(boxes_right, real_bboxes_repeated, P_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)

    # # consider iff. S/T/O/P appears inside the octagon. no order
    # s_class_score = compute_score(real_bboxes, real_bboxes_repeated, S_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)
    # t_class_score = compute_score(real_bboxes, real_bboxes_repeated, T_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)
    # o_class_score = compute_score(real_bboxes, real_bboxes_repeated, O_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)
    # p_class_score = compute_score(real_bboxes, real_bboxes_repeated, P_CLASS_INT, pred_repeated, pred_unsqueezed, num_of_boxes)

    # stop_score = s_class_score * t_class_score * o_class_score * p_class_score
    # stop_score = (s_class_score + t_class_score + o_class_score + p_class_score)
    stop_score = torch.log(s_class_score + const_noise) + torch.log(t_class_score + const_noise) + torch.log(o_class_score + const_noise) + torch.log(p_class_score + const_noise)

    print('stop_score.shape', stop_score.shape)

    tmp_shape = prediction.shape
    tmp = stop_score.view(tmp_shape[0], tmp_shape[1], tmp_shape[2], tmp_shape[3])
    print('tmp.shape', tmp.shape)

    # TODO: Try addition of values before sigmoid?
    # print(stop_score)
    # prediction[...,-1] *= stop_score
    # prediction[...,-1] = (prediction[...,-1] + stop_score) / 5
    print('prediction[...,-1].shape', prediction[...,-1].shape)
    prediction[...,-1] = torch.log(prediction[...,-1] + const_noise) + stop_score.view(tmp_shape[0], tmp_shape[1], tmp_shape[2], tmp_shape[3])

    return prediction

