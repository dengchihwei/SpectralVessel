# -*- coding = utf-8 -*-
# @File Name : spectral
# @Date : 2023/6/6 17:32
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import time
import torch
import scipy
import dataset
import argparse
import numpy as np
import torch.nn.functional as F
from train import read_json
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader


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


def get_diagonal(w: scipy.sparse.csr_matrix, threshold: float = 1e-12):
    d = w.dot(np.ones(w.shape[1], w.dtype))
    d[d < threshold] = 1.0  # Prevent division by zero.
    d = scipy.sparse.diags(d)
    return d


def extract_eigs(config_file, features_folder, split='train', K=5, normalize=True, drop_neg=True):
    start = time.time()
    config = read_json(config_file)
    # define the dataloader
    data_loader = DataLoader(getattr(dataset, config[split]['type'])(**config[split]['args']), batch_size=1)
    # get features
    print('Retrieving the features...')
    sem_feature = torch.load(os.path.join(features_folder, '{}-sem-0-4.pt'.format(split)), map_location='cpu')[:, :64, :64]
    dir_feature = torch.load(os.path.join(features_folder, '{}-dir-0-4.pt'.format(split)), map_location='cpu')[:, :64, :64]
    rad_feature = torch.load(os.path.join(features_folder, '{}-rad-0-4.pt'.format(split)), map_location='cpu')[:, :64, :64]
    sem_feature = sem_feature.contiguous().view(sem_feature.size(0), -1).T
    dir_feature = dir_feature.contiguous().view(dir_feature.size(0), -1).T
    rad_feature = rad_feature.contiguous().view(rad_feature.size(0), -1).T
    # normalize the features
    if normalize:
        sem_feature = F.normalize(sem_feature, p=2, dim=0)
        dir_feature = F.normalize(dir_feature, p=2, dim=0)
        rad_feature = F.normalize(rad_feature, p=2, dim=0)
    # compute affinities
    print('Computing affinities...')
    w_sem = sem_feature @ sem_feature.T
    w_dir = dir_feature @ dir_feature.T
    w_rad = rad_feature @ rad_feature.T
    if drop_neg:
        w_sem = w_sem * (w_sem > 0)
        w_dir = w_dir * (w_dir > 0)
        w_rad = w_rad * (w_rad > 0)
    # diagonal
    print('Computing diagonal...')
    w_comb = (1.0 * w_sem + 0.0 * w_dir + 0.0 * w_rad)
    w_comb = w_comb.numpy()
    d_comb = np.array(get_diagonal(w_comb).todense())
    # compute eigenvectors
    print('Computing eigen decomposition...')
    try:
        eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(d_comb - w_comb, k=(K + 1), sigma=0, which='LM', M=d_comb)
    except:
        print('Failed with LM, trying SM...')
        eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(d_comb - w_comb, k=(K + 1), which='SM', M=d_comb)
    eig_vals = torch.from_numpy(eig_vals)
    eig_vecs = torch.from_numpy(eig_vecs.T).float()
    # resolve sign ambiguity
    for k in range(eig_vecs.shape[0]):
        if 0.5 < torch.mean((eig_vecs[k] > 0).float()).item() < 1.0:  # reverse segment
            eig_vecs[k] = 0 - eig_vecs[k]
    print(eig_vecs.size(), eig_vals.size())


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/drive/adaptive_lc.json')
parser.add_argument('-f', '--feature_folder', type=str, default='../features/2023-06-08/ADAPTIVE_LC')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    extract_eigs(args.config_file, args.feature_folder)

