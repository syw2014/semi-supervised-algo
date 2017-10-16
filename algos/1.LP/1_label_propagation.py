#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
# Author: Jerry.Shi
# Date: 2017-10-16

# Description:
#   Label propagation is a semi-supervised learning algorithms, we can be used to find data with labels,and expand labeled data.
#   reference: http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf

import numpy as np
import math

def knn_nodes(dataset, query, k):
    """Only build a graph which node only connect the k nearest neighbors. This function return k nearest nodes based on
    Euclidean distance.
    - dataset: the total dataset matrix
    - return: k-nearest sample indices"""
    num_sample = dataset.shape[0]
    diff = np.tile(query, (num_sample, 1)) - dataset
    dist = diff ** 2
    dist = np.sum(dist, axis = 1)

    sorted_dist_indices = np.argsort(dist)
    if k > len(sorted_dist_indices):
        k = len(sorted_dist_indices)
    return sorted_dist_indices[: k]


def build_graph(matrix, k=None):
    """Build weight connect graph.
    - matrix: total dataset matrix, shape=N*C, N is the number of samples, C is the number of labels.
    - return: transition weight matrix"""
    num_samples = matrix.shape[0]
    trans_matrix = np.zeros((num_samples, num_samples), np.float32)
    assert k > 0,("Please specific the k for Knn")
    for i in xrange(num_samples):
        k_nodes = knn_nodes(matrix, matrix[i, :], k)
        trans_matrix[i][k_nodes] = 1.0 / k
    return trans_matrix


def label_propagation(label_mat, unlabel_mat, labels, k=10, max_iter=500, tol=1e-3):
    """Label propagation algorithm.
    - label_mat, labeled data with a shape = L * C, L is the number of labeled data , C is the number of label
    - unlabel_mat, unlabeled data with shape = Y * C, Y is the number of unlabeled data, C is the number of label
    - labels, list of label
    - k, the number of nearest neighbour
    - max_iter, maximum iteration times"""
    num_label_samples = label_mat.shape[0]
    num_unlabel_samples = unlabel_mat.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    num_labels = len(np.unique(labels))

    matrix = np.vstack((label_mat, unlabel_mat))
    fixed_data = np.zeros((num_label_samples, num_labels), np.float32)
    for i in xrange(num_label_samples):
        fixed_data[i][labels[i]] = 1.0

    label_function = np.zeros((num_samples, num_labels),  np.float32)
    label_function[0: num_label_samples] = fixed_data
    label_function[num_label_samples: num_samples] = -1

    # build graph
    trans_matrix = build_graph(matrix, k)

    # start propagation
    iters = 0
    pre_label_function = np.zeros((num_samples, num_labels), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()
    while iters < max_iter and changed > tol:
        if iters % 1  == 0:
            print("Iteration:{}/{} changed:{:.3f}".format(iters,max_iter, changed))
            pre_label_function = label_function
            iters += 1

            # propagation
            label_function = np.dot(trans_matrix, label_function)

            # clamp
            label_function[0: num_label_samples] = fixed_data

            # check converage
            changed = np.abs(pre_label_function - label_function).sum()
    # get terminate label of unlabeled 
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in xrange(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i+num_label_samples])
    return unlabel_data_labels


def create_data(num_data):
    center = np.array([5.0, 5.0])
    radius_inner = 2
    radius_outer = 4
    num_inner = num_data / 3
    num_outer = num_data - num_inner

    data = []
    theta = 0.0
    for i in xrange(num_inner):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radius_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radius_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 2

    theta = 0.0
    for i in xrange(num_outer):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radius_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radius_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 1

    label_mat = np.zeros((2,2), np.float32)
    label_mat[0] = center + np.array([-radius_inner+0.5, 0])
    label_mat[1] = center + np.array([-radius_outer+0.5, 0])
    labels = [0, 1]
    unlabel_mat = np.vstack(data)
    return label_mat, labels, unlabel_mat


if __name__ == "__main__":
    num_data = 800
    label_mat, labels, unlabel_mat = create_data(num_data)

    unlabel_data_labels = label_propagation(label_mat, unlabel_mat, labels)
