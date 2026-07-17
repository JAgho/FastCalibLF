import numpy as np

def read_data(data, readout_length, n_averages, coords, dummies):
    real_data = data
    data_reshape = np.reshape(real_data, (coords, n_averages, 3, readout_length))
    return data_reshape

def read_data_spgr_b1(raw, n_repetitions, n_readout):
    raw_flat = np.asarray([item.processed_data[0] for item in raw])
    data = np.reshape(raw_flat, (n_repetitions, 2, n_readout))
    data_avg = np.mean(data, axis=0)

    return data_avg

    
    
