import torch
import numpy as np
from loss_fns import pairwise_l2_dist

def sum_pow_mat(A, max_n = 15):
    A = A.float()
    S = A.clone().float()
    for i in range(max_n): # lots of allocation...
        S += S.matmul(A)
        S.sign_()
    return S

def fast_batch_cwa(points, weights, pre_weight_thresh, d_thresh, post_weight_thresh, n_steps = 10, theta_mul = None):
    if theta_mul is not None:
        print(points.shape)
        points *= theta_mul.view(1,1,-1)

    weights[weights < pre_weight_thresh] = 0.0
    threshold = d_thresh
    A = (pairwise_l2_dist(points) < threshold)
    #### zero out rows and columns corresponding to zero-weight points.
    zero_inds = ((weights == 0).unsqueeze(-1).float()*(torch.ones_like(weights).unsqueeze(-2))).sign_().byte()
    A[zero_inds] = 0
    zero_inds = ((weights == 0).unsqueeze(-2).float()*(torch.ones_like(weights).unsqueeze(-1))).sign_().byte()
    A[zero_inds] = 0
    # Compute connected components using power iteration
    C = sum_pow_mat(A, n_steps)
    weighted_C = C*weights.float().unsqueeze(-2)
    C_weights = weighted_C.sum(-1)
    C_means = (weighted_C.matmul(points.float())/C_weights.unsqueeze(-1))
    C_means = C_means.cpu().numpy()
    C_weights[C_weights<post_weight_thresh] = 0.0
    C_weights = C_weights.cpu().numpy()
    all_points = []
    all_weights = []
    for b_idx in range(points.shape[0]):
        mask = np.full(points.shape[1], True, dtype=bool)
        # This assumes float arithmetic works perfectly. Fun fact: it does not.
        (ws,inds) = np.unique(np.round(C_weights[b_idx],4),True)
        #assert(ws[0] == 0)
        all_points.append(C_means[b_idx,inds[1:],:])
        all_weights.append(C_weights[b_idx,inds[1:]])
    if theta_mul is not None:
        theta_mul = theta_mul.cpu().view(1,-1).numpy()
        all_points = [p/theta_mul for p in all_points]
    return all_points, all_weights
