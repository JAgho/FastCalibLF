"""Derive first-order (linear) B0 shim offsets from a low-res 3D double-echo phase map.

data (echo, x, y, z) complex -> phase-difference field map (Hz) -> 3D unwrap ->
magnitude-weighted linear fit -> per-axis shim offset in mT/m.

Only first-order (linear) shim is derived; gradient-offset shimming can only
produce a field linear in position, so residual higher-order inhomogeneity is
left untouched.
"""
import numpy as np

# Proton gyromagnetic ratio / 2pi (Hz/T). Independent of B0 field strength.
GAMMA_BAR_HZ_PER_T = 42.577478518e6


def linear_shim_from_fieldmap(data: np.ndarray,
                              mask: np.ndarray,
                              voxel_size_m,
                              delta_te: float) -> dict:
    """Compute linear shim offsets (mT/m) from a double-echo 3D acquisition.

    Parameters
    ----------
    data
        Complex array of shape (echo, x, y, z). data[0] and data[1] are the
        two gradient echoes at TE1 and TE2.
    mask
        Boolean array of shape (x, y, z), True where signal is trusted. Used
        both to confine the 3D unwrap and to select voxels for the fit.
    voxel_size_m
        (dx, dy, dz) voxel size in metres (e.g. FOV_axis / N_axis per axis).
    delta_te
        TE2 - TE1 in seconds.

    Returns
    -------
    dict
        {"x_mt": ..., "y_mt": ..., "z_mt": ...}  shim offsets in mT/m to APPLY
        (already negated to cancel the measured gradient). Keys match
        set_shim_offsets so you can call set_shim_offsets(**shim). Also includes
        "residual_f0_hz".
    """
    from skimage.restoration import unwrap_phase

    echo1, echo2 = data[0], data[1]
    magnitude = np.abs(echo1)

    # Wrapped phase difference over delta_te; unwrap in 3D, confined to the mask
    # so noise voxels can't seed spurious wrap paths.
    dphi = np.angle(echo2 * np.conj(echo1))
    dphi_uw = np.ma.filled(unwrap_phase(np.ma.array(dphi, mask=~mask)), 0.0)
    field_hz = dphi_uw / (2.0 * np.pi * delta_te)

    # Physical coordinate grids (metres) centred on isocentre, per axis.
    nx, ny, nz = field_hz.shape
    dx, dy, dz = voxel_size_m
    X, Y, Z = np.meshgrid((np.arange(nx) - (nx - 1) / 2) * dx,
                          (np.arange(ny) - (ny - 1) / 2) * dy,
                          (np.arange(nz) - (nz - 1) / 2) * dz,
                          indexing="ij")

    # Magnitude-weighted (inverse phase-noise variance ~ SNR^2) linear fit:
    #   field(r) = a + gx*x + gy*y + gz*z
    m = mask.ravel()
    A = np.stack([np.ones(X.size), X.ravel(), Y.ravel(), Z.ravel()], axis=1)[m]
    y = field_hz.ravel()[m]
    sw = magnitude.ravel()[m]                       # sqrt(weight); weight = mag^2
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)

    const_hz = beta[0]                              # residual bulk off-resonance
    grad_hz_per_m = beta[1:4]                        # field gradient (Hz/m) per axis

    # Shim to CANCEL the gradient: G_shim = -g / gamma_bar (T/m) -> *1e3 (mT/m).
    shim_mt = -grad_hz_per_m / GAMMA_BAR_HZ_PER_T * 1e3

    return {"x_mt": float(shim_mt[0]), "y_mt": float(shim_mt[1]), "z_mt": float(shim_mt[2]),
            "residual_f0_hz": float(const_hz)}