# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    # batch_heatmaps shape = (1, 17 ,96, 72)
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]                                                     # 72 -> heatmaps width

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))            # (1, 17, 6912)

    idx = np.argmax(heatmaps_reshaped, 2)                                               # np.argmax()获取array的某一个维度中数值最大的那个元素的索引 -> 代表了坐标
    maxvals = np.amax(heatmaps_reshaped, 2)                                             # np.amax()获取array的某一个维度中数值最大的那个元素 -> 代表了概率多大

    idx = idx.reshape((batch_size, num_joints, 1))                                      # idx -> (1, 17, 1)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))                              # maxvals -> max values -> (1, 17, 1)

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)                                  # preds -> (1, 17, 2) idx最后一维复制了一遍
    preds[:, :, 0] = (preds[:, :, 0]) % width                                           # pred[:, :, 0] / width 余数 -> 6912中最大值索引 / 72 -> heatmap中的高
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)                                 # pred[:, :, 1] / width 除的最大整数，向下取整         -> heatmap中的宽
    # np.floor() -> 返回不大于输入参数的最大整数

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))                            # pred_mask -> (1, 17, 2)
    pred_mask = pred_mask.astype(np.float32)                                            # pred_mask -> (True -> 1, False -> 0)

    preds *= pred_mask                                                                  # (1, 17, 2)

    return preds, maxvals


def get_max_preds_heatmap(batch_heatmaps, original_img, save_img_count):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    # batch_heatmaps shape = (1, 17 ,96, 72)
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]                                                     # 72 -> heatmaps width

    if save_img_count == 60:                                                            # which frame you want to see the heatmap
        plt.imshow(cv2.resize(original_img, (width, height)))
        plt.imshow(batch_heatmaps[0][0], cmap='viridis', interpolation='nearest', alpha=0.7)
        plt.axis('off')
        plt.colorbar()  # 添加颜色条
        plt.savefig("/home/ligaoqi/projects/python_projects/HRNet_With_Visualized_Heatmap/output/heatmap.jpg")
        print("heatmap图片已保存")

    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))            # (1, 17, 6912)

    idx = np.argmax(heatmaps_reshaped, 2)                                               # np.argmax()获取array的某一个维度中数值最大的那个元素的索引 -> 代表了坐标
    maxvals = np.amax(heatmaps_reshaped, 2)                                             # np.amax()获取array的某一个维度中数值最大的那个元素 -> 代表了概率多大

    idx = idx.reshape((batch_size, num_joints, 1))                                      # idx -> (1, 17, 1)
    maxvals = maxvals.reshape((batch_size, num_joints, 1))                              # maxvals -> max values -> (1, 17, 1)

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)                                  # preds -> (1, 17, 2) idx最后一维复制了一遍
    preds[:, :, 0] = (preds[:, :, 0]) % width                                           # pred[:, :, 0] / width 余数 -> 6912中最大值索引 / 72 -> heatmap中的高
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)                                 # pred[:, :, 1] / width 除的最大整数，向下取整         -> heatmap中的宽
    # np.floor() -> 返回不大于输入参数的最大整数

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))                            # pred_mask -> (1, 17, 2)
    pred_mask = pred_mask.astype(np.float32)                                            # pred_mask -> (True -> 1, False -> 0)

    preds *= pred_mask                                                                  # (1, 17, 2)

    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):                                                # n -> 第n个人
            for p in range(coords.shape[1]):                                            # p -> 第p个关节点
                hm = batch_heatmaps[n][p]                                               # batch_heatmaps shape = (1, 17 ,96, 72)
                px = int(math.floor(coords[n][p][0] + 0.5))                             # math.floor() 函数用来返回数字的下舍整数，即它总是将数值向下舍入为最接近的整数
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],                                # hm -> (96, 72)
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )                                                                   # diff -> heatmap关节点左右差值, heatmap关节点
                    coords[n][p] += np.sign(diff) * .25                                 # sign()是Python的Numpy中的取数字符号(数字前的正负号)的函数

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], [heatmap_width, heatmap_height])

    return preds, maxvals


def get_final_preds_heatmap(config, batch_heatmaps, center, scale, original_img, save_img_count):
    coords, maxvals = get_max_preds_heatmap(batch_heatmaps, original_img, save_img_count)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):                                                # n -> 第n个人
            for p in range(coords.shape[1]):                                            # p -> 第p个关节点
                hm = batch_heatmaps[n][p]                                               # batch_heatmaps shape = (1, 17 ,96, 72)
                px = int(math.floor(coords[n][p][0] + 0.5))                             # math.floor() 函数用来返回数字的下舍整数，即它总是将数值向下舍入为最接近的整数
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],                                # hm -> (96, 72)
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )                                                                   # diff -> heatmap关节点左右差值, heatmap关节点
                    coords[n][p] += np.sign(diff) * .25                                 # sign()是Python的Numpy中的取数字符号(数字前的正负号)的函数

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], [heatmap_width, heatmap_height])

    return preds, maxvals
