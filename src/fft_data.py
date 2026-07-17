import numpy as np

def fft_data(data):
    return np.fft.fftshift(np.fft.fftn(data, axes=[3]), 3)
