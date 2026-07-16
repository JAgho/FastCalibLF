def fft_data(data):
    return np.fft.fftshift(np.fft.fft(data, 3), 3)
