def fit_alpha(data_avg):
    s1 = np.abs(data_avg[0])
    s2 = np.abs(data_avg[1])
    n = 5
    r = s2[5] / s1[5]
    cos_theta = (n * r - 1) / (n - r)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_deg = np.rad2deg(np.arccos(cos_theta))

    return theta_deg
