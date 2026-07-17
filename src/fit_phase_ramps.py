import numpy as np
from scipy.optimize import least_squares

def fit_phase_ramp(phi, mask, dTE, dx=1.0):
    """Return (B0 slope, B0 offset) in Hz/dx and Hz from wrapped phase."""
    x = (np.arange(phi.size) - (phi.size - 1) / 2) * dx
    x, y = x[mask], phi[mask]

    p0 = np.polyfit(x, np.unwrap(y), 1)
    slope, offset = least_squares(
        lambda p: np.angle(np.exp(1j * (y - p[0]*x - p[1]))),
        p0,
    ).x

    return slope / (2*np.pi*dTE), offset / (2*np.pi*dTE)
