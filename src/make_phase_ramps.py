def make_phase_ramps(hybrid):
    av_ramps = mean(hybrid, axis=(1))
    r1 = av_ramps[:, 1, :]
    r2 = av_ramps[:, 2, :]
    diff = np.angle(r1 * np.conj(r2))
    hshape = av_ramps.shape
    unwrapped_phase = zeros(1, hshape[1:-1], dtype=complex)
    for coord in range(3):
        unwrapped_phase[0, :,:] = np.unwrap(diff[coord, :], axis=-1)
    return unwrapped_phase

