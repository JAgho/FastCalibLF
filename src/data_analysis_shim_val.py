import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input data and acquisition parameters
# ============================================================
data = np.load("src/hybrid.npy")
print("Raw data:", data.shape)

axis_names = ("X", "Y", "Z")

fov = 220e-3
te_1 = 6e-3
te_2 = 8e-3
delta_te = te_2 - te_1

tr_2_factor = 5
snr_threshold = 2

gamma_hz = 42.57747892e6
gamma_rad = 2 * np.pi * gamma_hz

theta_all = []
fa_means = []
phase_results = []
shim_values_mt_m = []

fig, ax = plt.subplots(2, 3, figsize=(15, 8))


for axis_idx, axis_name in enumerate(axis_names):
    data_ax = data[axis_idx]
    print(f"{axis_name} axis:", data_ax.shape)

    # ============================================================
    # Reverse Echo 2
    # ============================================================
    data_ax_corrected = data_ax.copy()
    data_ax_corrected[:, 2, :] = np.flip(data_ax_corrected[:, 2, :],axis=-1,)

    data_avg_corrected = np.mean(data_ax_corrected, axis=0)
    print(f"{axis_name} axis avg:", data_avg_corrected.shape)

    # ============================================================
    # SNR mask
    # ============================================================
    noise_std = np.std(data_ax_corrected[:, 0, :],axis=0,)

    snr = (np.abs(data_avg_corrected[0])/ (noise_std + 1e-12))

    mask = snr >= snr_threshold

    # ============================================================
    # B1 calibration
    # ============================================================
    s1 = np.abs(data_avg_corrected[0])
    s2 = np.abs(data_avg_corrected[1])

    valid_fa = (mask & (s1 > 0) & (np.abs(tr_2_factor * s1 - s2) > 1e-12))

    theta_deg = np.full(s1.shape, np.nan, dtype=float,)

    ratio = s2[valid_fa] / s1[valid_fa]

    cos_theta = (tr_2_factor * ratio - 1) / (tr_2_factor - ratio)

    physical = ((cos_theta >= -1) & (cos_theta <= 1))

    valid_indices = np.where(valid_fa)[0]

    theta_deg[valid_indices[physical]] = np.rad2deg(np.arccos(cos_theta[physical]))

    fa_mean = np.nanmean(theta_deg)

    theta_all.append(theta_deg)
    fa_means.append(fa_mean)

    # ============================================================
    # Phase difference
    # ============================================================
    phase_per_repetition = np.angle(data_ax_corrected[:, 2, :]* np.conjugate(data_ax_corrected[:, 1, :]))

    print(f"{axis_name} phase:",phase_per_repetition.shape,)

    phase_diff = np.angle(np.mean(np.exp(1j * phase_per_repetition),axis=0,))

    print(f"{axis_name} phase avg:",phase_diff.shape,)

    phase_unwrapped = np.unwrap(phase_diff)
    phase_unwrapped[~mask] = np.nan

    # ============================================================
    # Spatial coordinates
    # ============================================================
    n_samples = phase_unwrapped.size
    spatial_resolution = fov / n_samples

    position_m = (np.arange(n_samples)- (n_samples - 1) / 2) * spatial_resolution

    valid_phase = np.isfinite(phase_unwrapped)

    # ============================================================
    # Linear phase fit and shim calculation
    # ============================================================
    if np.count_nonzero(valid_phase) >= 2:
        slope_rad_m, intercept = np.polyfit(position_m[valid_phase],phase_unwrapped[valid_phase],1,)

        phase_fit = (slope_rad_m * position_m + intercept)

        measured_gradient_t_m = (slope_rad_m / (gamma_rad * delta_te))

        measured_gradient_mt_m = (measured_gradient_t_m * 1e3)

        shim_gradient_mt_m = (-measured_gradient_mt_m)

    else:
        slope_rad_m = np.nan
        intercept = np.nan

        phase_fit = np.full(position_m.shape,np.nan,dtype=float,)

        measured_gradient_mt_m = np.nan
        shim_gradient_mt_m = np.nan

    shim_values_mt_m.append(shim_gradient_mt_m)

    phase_results.append(
        {
            "phase": phase_unwrapped,
            "fit": phase_fit,
            "slope_rad_m": slope_rad_m,
            "intercept": intercept,
            "measured_gradient_mt_m": measured_gradient_mt_m,
            "shim_gradient_mt_m": shim_gradient_mt_m,
        }
    )

    print(
        f"{axis_name} measured gradient: "
        f"{measured_gradient_mt_m:.6f} mT/m"
    )

    print(
        f"{axis_name} required shim gradient: "
        f"{shim_gradient_mt_m:.6f} mT/m"
    )

    # ============================================================
    # Flip-angle plot
    # ============================================================
    ax[0, axis_idx].plot(position_m * 1e3,theta_deg,"o",)

    ax[0, axis_idx].axhline(fa_mean,linestyle="--",label=f"Mean = {fa_mean:.2f}°",)

    ax[0, axis_idx].set_title(f"{axis_name} projection")

    ax[0, axis_idx].set_xlabel("Position [mm]")

    ax[0, axis_idx].set_ylabel("Flip angle [deg]")

    ax[0, axis_idx].grid(True)
    ax[0, axis_idx].legend()

    # ============================================================
    # Phase-difference plot
    # ============================================================
    ax[1, axis_idx].plot(position_m * 1e3,phase_unwrapped,label="Phase difference",)

    ax[1, axis_idx].plot(
        position_m * 1e3, phase_fit, "--",
        label=(f"Shim = "f"{shim_gradient_mt_m:.4f} mT/m"),)

    ax[1, axis_idx].set_title(f"{axis_name} projection")

    ax[1, axis_idx].set_xlabel("Position [mm]")

    ax[1, axis_idx].set_ylabel("Phase [rad]")

    ax[1, axis_idx].grid(True)
    ax[1, axis_idx].legend()


# ============================================================
# Global results
# ============================================================
fa_global = np.nanmean(fa_means)

shim_x, shim_y, shim_z = shim_values_mt_m

fig.suptitle(
    "FAST CALIBRATION\n"
    f"FA mean = {fa_global:.2f}°\n"
    f"Shim values [mT/m] = "
    f"[{shim_x:.4f}, {shim_y:.4f}, {shim_z:.4f}]"
)

fig.tight_layout(rect=[0, 0, 1, 0.91])
plt.show()


# ============================================================
# Print final calibration values
# ============================================================
print("\nFinal calibration results")
print(f"Mean flip angle: {fa_global:.2f} deg")
print(f"X shim: {shim_x:.6f} mT/m")
print(f"Y shim: {shim_y:.6f} mT/m")
print(f"Z shim: {shim_z:.6f} mT/m")