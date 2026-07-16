import numpy as np

def mask_projection_by_snr(proj, percentile=20):
    avs, rlen = proj[:, -1, :].shape
    stds = np.std(proj, axis=0)
    ptile = np.percentile(stds, percentile)
    mask = stds > ptile
    return mask

def mask_data(data, mask_x, mask_y, mask_z):
    masked_data = np.copy(data)
    masked_data[0,...] *= mask_x[np.newaxis, :, :]
    masked_data[1,...] *= mask_y[np.newaxis, :, :]
    masked_data[2,...] *= mask_z[np.newaxis, :, :]
    return masked_data
