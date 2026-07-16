import numpy as np

def read_data(data, readout_length, n_averages, coords, dummies):
    real_data = data[0:dummies-1]
    data_reshape = np.reshape(real_data, (coords, n_averages, 2, readout_length))
    return data_reshape
    
    
