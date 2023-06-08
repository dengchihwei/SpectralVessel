# -*- coding = utf-8 -*-
# @File Name : spectral
# @Date : 2023/6/6 17:32
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import scipy
import numpy as np
from sklearn.neighbors import KDTree


def normalize_rows(A, threshold=0.0):
    row_sums = A.dot(np.ones(A.shape[1], A.dtype))
    # Prevent division by zero.
    row_sums[row_sums < threshold] = 1.0
    row_normalization_factors = 1.0 / row_sums
    D = scipy.sparse.diags(row_normalization_factors)
    A = D.dot(A)
    return A


def knn_affinity(image, n_neighbors=(20, 10), distance_weights=(2.0, 0.1), kernel="soft"):
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    # get the image coordinates
    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)
    # Store weight matrix indices and values in sparse coordinate form.
    i, j, coo_data = [], [], []
    for k, distance_weight in zip(n_neighbors, distance_weights):
        # Features consist of RGB color values and weighted spatial coordinates.
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((h * w, 5), dtype=np.float32),
        )
        # Find indices of nearest neighbors in feature space.
        # _, neighbor_indices = knn(f, f, k=k)
        _, neighbor_indices = KDTree(f, leaf_size=8).query(f, k=k)

        # [0 0 0 0 0 (k times) 1 1 1 1 1 2 2 2 2 2 ...]
        i.append(np.repeat(np.arange(h * w), k))
        j.append(neighbor_indices.ravel())
        w_ij = np.ones(k * h * w)
        if kernel == "soft":
            w_ij -= np.abs(f[i[-1]] - f[j[-1]]).sum(axis=1) / f.shape[1]
        coo_data.append(w_ij)
    # Add matrix to itself to get a symmetric matrix.
    # The '+' here is list concatenation and not addition.
    # The csr_matrix constructor will do the addition later.
    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.concatenate(coo_data + coo_data)
    # Assemble weights from coordinate sparse matrix format.
    w = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (h * w, h * w))
    # make sure the matrix is symmetric
    w = normalize_rows(w) / 2
    w = w + w.T
    return w

