import numpy as np

def make_magnitude_max(data):
    maxima = np.argmax(data, axis=-1)
    maximal_r = data[:,:,:,maxima]
    mean_r = np.mean(maximal_r, 1)
    return mean_r

def make_magnitude_mean(data):
    mean = np.mean(data, axis=-1)
    return mean
