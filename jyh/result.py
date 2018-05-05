#!/usr/bin/env python
# -*- coding:utf-8 -*-

import scipy.io as io
import numpy as np


# from sklearn import mixture
# import torch as np
def DETsort(x, col=''):
    # DETsort Sort rows

    assert x.ndim > 1, 'x must be a 2D matrix'
    # if col == '':
    #     list(range(1, x.shape[1]))

    ndx = np.arange(x.shape[0])

    # sort 2nd column ascending
    ind = np.argsort(x[:, 1])
    ndx = ndx[ind]

    # reverse to descending order
    ndx = ndx[::-1]

    # now sort first column ascending
    ind = np.argsort(x[ndx, 0])

    ndx = ndx[ind]
    sort_scores = x[ndx, :]
    return sort_scores


def Compute_DET(targetScore, notargetScore):
    num_true = targetScore.size
    num_false = notargetScore.size
    total = num_true + num_false
    Pmiss = np.zeros([total + 1, 1], 'float32')
    Pfa = np.zeros([total + 1, 1], 'float32')

    scores = np.zeros([total, 2], 'float32')
    scores[0:num_false, 0] = notargetScore
    scores[0:num_false, 1] = 0
    scores[num_false:total, 0] = targetScore
    scores[num_false:total, 1] = 1

    scores = DETsort(scores)
    sumtrue = np.cumsum(scores[:, 1], 0)
    sumfalse = num_false - (np.arange(total) + 1 - sumtrue)

    Pmiss[0] = 0
    Pfa[0] = 1.0
    Pmiss[1:] = sumtrue.reshape(sumtrue.size, 1) / num_true
    Pfa[1:] = sumfalse.reshape(sumfalse.size, 1) / num_false
    abs_def = np.abs(Pmiss - Pfa)
    # [minDes, EER_pos] = np.min(abs_def, axis=0)
    EER_pos = np.argmin(abs_def, axis=0)
    minDes = abs_def[EER_pos]
    return Pmiss, Pfa


def cal_eer(Score, Key, NumClass):
    # score :nxNumClass
    class_id = range(NumClass)
    id = Key == class_id
    id = id.transpose()
    Score = Score.transpose()
    targetScore = Score[id]
    notargetScore = Score[id == False]

    [Pmiss, Pfa] = Compute_DET(targetScore, notargetScore)
    # out
    P_gap = np.abs(Pmiss - Pfa)
    minIdx = np.argmin(P_gap, axis=0)
    minDes = P_gap[minIdx]
    eer = (Pmiss[minIdx] + Pfa[minIdx]) / 2
    return eer[0]


def cal_eer2(scores, key):
    targetScore, notargetScore = scores.get_tar_non(key)
    [Pmiss, Pfa] = Compute_DET(targetScore, notargetScore)
    # out
    P_gap = np.abs(Pmiss - Pfa)
    minIdx = np.argmin(P_gap, axis=0)
    minDes = P_gap[minIdx]
    eer = (Pmiss[minIdx] + Pfa[minIdx]) / 2
    return eer[0]
