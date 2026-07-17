import numpy as np
import matplotlib.pyplot as plt


data = np.load("src/hybrid_60.npy")
print("Raw data:", data.shape)

axis_names = ("X", "Y", "Z")
tr_2_factor = 5
snr_threshold = 2

theta_all = []
phase_results = []
fa_means = []

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

for axis_idx, axis_name in enumerate(axis_names):
    data_ax = data[axis_idx]
    print(f"{axis_name} axis:", data_ax.shape)

    # # Average repetitions
    # data_avg = np.mean(data_ax, axis=0)
    # print(data_avg.shape)

    # Reverse Echo 2 
    data_ax_corrected = data_ax.copy()
    data_ax_corrected[:, 2, :] = np.flip(data_ax_corrected[:, 2, :], axis=-1)

    data_avg_corrected = np.mean(data_ax_corrected, axis=0)
    print(f"{axis_name} axis avg:", data_avg_corrected.shape)

    # ============================================================
    # SNR mask
    # ============================================================
    noise_std = np.std(data_ax_corrected[:, 0, :], axis=0)
    snr = np.abs(data_avg_corrected[0]) / (noise_std + 1e-12)
    mask = snr >= snr_threshold

    # ============================================================
    # B1 calibration
    # ============================================================
    s1 = np.abs(data_avg_corrected[0])
    s2 = np.abs(data_avg_corrected[1])

    valid_fa = (
        mask
        & (s1 > 0)
        & (np.abs(tr_2_factor * s1 - s2) > 1e-12)
    )

    theta_deg = np.full(s1.shape, np.nan, dtype=float)

    ratio = s2[valid_fa] / s1[valid_fa]
    cos_theta = (tr_2_factor * ratio - 1) / (tr_2_factor - ratio)

    physical = (cos_theta >= -1) & (cos_theta <= 1)

    valid_indices = np.where(valid_fa)[0]
    theta_deg[valid_indices[physical]] = np.rad2deg(
        np.arccos(cos_theta[physical])
    )

    fa_mean = np.nanmean(theta_deg)

    theta_all.append(theta_deg)
    fa_means.append(fa_mean)

    # ============================================================
    # Phase difference and linear fit
    # ============================================================
    phase_per_repetition = np.angle(
        data_ax_corrected[:, 2, :]
        * np.conjugate(data_ax_corrected[:, 1, :])
    )
    print(f"{axis_name} phase:", phase_per_repetition.shape)

    phase_diff = np.angle(
        np.mean(np.exp(1j * phase_per_repetition), axis=0)
    )
    print(f"{axis_name} phase avg:", phase_diff.shape)

    phase_unwrapped = np.unwrap(phase_diff)
    phase_unwrapped[~mask] = np.nan

    x_phase = np.arange(phase_unwrapped.size)
    valid_phase = np.isfinite(phase_unwrapped)

    if np.count_nonzero(valid_phase) >= 2:
        slope, intercept = np.polyfit(
            x_phase[valid_phase],
            phase_unwrapped[valid_phase],
            1,
        )
        phase_fit = slope * x_phase + intercept
    else:
        slope = np.nan
        intercept = np.nan
        phase_fit = np.full_like(x_phase, np.nan, dtype=float)

    phase_results.append(
        {
            "phase": phase_unwrapped,
            "fit": phase_fit,
            "slope": slope,
            "intercept": intercept,
        }
    )

    # ============================================================
    # Flip-angle plot
    # ============================================================
    ax[0, axis_idx].plot(theta_deg, "o")
    ax[0, axis_idx].axhline(
        fa_mean,
        linestyle="--",
        label=f"Mean = {fa_mean:.2f}°",
    )
    ax[0, axis_idx].set_title(f"{axis_name} projection")
    ax[0, axis_idx].set_ylabel("Flip angle [deg]")
    ax[0, axis_idx].grid(True)
    ax[0, axis_idx].legend()

    # ============================================================
    # Phase-difference plot
    # ============================================================
    ax[1, axis_idx].plot(
        x_phase,
        phase_unwrapped,
        label="Phase difference",
    )
    ax[1, axis_idx].plot(
        x_phase,
        phase_fit,
        "--",
        label=f"Fit: y = {slope:.4f}x + {intercept:.4f}",
    )
    ax[1, axis_idx].set_title(f"{axis_name} projection")
    ax[1, axis_idx].set_xlabel("Sample")
    ax[1, axis_idx].set_ylabel("Phase [rad]")
    ax[1, axis_idx].grid(True)
    ax[1, axis_idx].legend()


fa_global = np.mean(fa_means)

fig.suptitle(
    f"FAST CALIBRATION\nFA mean = {fa_global:.2f}°\n Shim values = [??,??,??]"
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
