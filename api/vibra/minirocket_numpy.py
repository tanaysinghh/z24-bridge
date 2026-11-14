from itertools import combinations

import numpy as np

KERNEL_INDICES = np.array(list(combinations(range(9), 3)), dtype=np.int64)


def transform(X, dilations, num_features_per_dilation, biases):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[:, 0, :]
    n, length = X.shape
    A = -X
    G = X + X + X
    out = np.zeros((n, len(KERNEL_INDICES) * int(np.sum(num_features_per_dilation))), dtype=np.float32)
    feature_start = 0
    for dilation_index, dilation in enumerate(np.asarray(dilations).tolist()):
        padding0 = dilation_index % 2
        padding = ((9 - 1) * dilation) // 2
        n_feat = int(num_features_per_dilation[dilation_index])
        c_alpha = A.copy()
        c_gamma = np.zeros((9, n, length), dtype=np.float32)
        c_gamma[4] = G
        start, end = dilation, length - padding
        for gamma_index in range(4):
            c_alpha[:, -end:] = c_alpha[:, -end:] + A[:, :end]
            c_gamma[gamma_index, :, -end:] = G[:, :end]
            end += dilation
        for gamma_index in range(5, 9):
            c_alpha[:, :-start] = c_alpha[:, :-start] + A[:, start:]
            c_gamma[gamma_index, :, :-start] = G[:, start:]
            start += dilation
        for kernel_index, (i0, i1, i2) in enumerate(KERNEL_INDICES):
            c = c_alpha + c_gamma[i0] + c_gamma[i1] + c_gamma[i2]
            if (padding0 + kernel_index) % 2 == 1:
                c = c[:, padding:-padding]
            b = biases[feature_start : feature_start + n_feat]
            out[:, feature_start : feature_start + n_feat] = (c[:, :, None] > b[None, None, :]).mean(axis=1)
            feature_start += n_feat
    return out


def transform_multi(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases):
    X = np.asarray(X, dtype=np.float32)
    n, n_ch, length = X.shape
    A = -X
    G = X + X + X
    out = np.zeros((n, len(KERNEL_INDICES) * int(np.sum(num_features_per_dilation))), dtype=np.float32)
    feature_start = 0
    combination = 0
    channel_start = 0
    for dilation_index, dilation in enumerate(np.asarray(dilations).tolist()):
        padding0 = dilation_index % 2
        padding = ((9 - 1) * dilation) // 2
        n_feat = int(num_features_per_dilation[dilation_index])
        c_alpha = A.copy()
        c_gamma = np.zeros((9, n, n_ch, length), dtype=np.float32)
        c_gamma[4] = G
        start, end = dilation, length - padding
        for gamma_index in range(4):
            c_alpha[:, :, -end:] = c_alpha[:, :, -end:] + A[:, :, :end]
            c_gamma[gamma_index, :, :, -end:] = G[:, :, :end]
            end += dilation
        for gamma_index in range(5, 9):
            c_alpha[:, :, :-start] = c_alpha[:, :, :-start] + A[:, :, start:]
            c_gamma[gamma_index, :, :, :-start] = G[:, :, start:]
            start += dilation
        for kernel_index, (i0, i1, i2) in enumerate(KERNEL_INDICES):
            k = int(num_channels_per_combination[combination])
            chans = np.asarray(channel_indices[channel_start : channel_start + k])
            c = (c_alpha[:, chans] + c_gamma[i0][:, chans] + c_gamma[i1][:, chans] + c_gamma[i2][:, chans]).sum(axis=1)
            if (padding0 + kernel_index) % 2 == 1:
                c = c[:, padding:-padding]
            b = biases[feature_start : feature_start + n_feat]
            out[:, feature_start : feature_start + n_feat] = (c[:, :, None] > b[None, None, :]).mean(axis=1)
            feature_start += n_feat
            combination += 1
            channel_start += k
    return out
