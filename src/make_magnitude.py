def make_magnitude(data):
    maxima = np.argmax(data, axis=3)
    maximal_r = data[:,:,:,maxima]
    mean_r = np.mean(maximal_r, 1)
    return mean_r
