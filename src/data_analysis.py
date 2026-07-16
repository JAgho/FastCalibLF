import numpy as np
import matplotlib.pyplot as plt

axis_proj = 'z'
data = np.load("src/hybrid.npy")

# axis_proj = 'z'
# data = np.load("src/hybrid_60.npy")
print('Raw:', data.shape)

if axis_proj == 'x':
    data_ax = data[0]
    fig_title = 'X projection'
elif axis_proj == 'y':
    data_ax = data[1]
    fig_title = 'Y projection'
elif axis_proj == 'z':
    data_ax = data[2]
    fig_title = 'Z projection'

print('One axis:', data_ax.shape)
data_avg = np.mean(data_ax, axis=0)
print('One axis - avg:', data_avg.shape)

dif_phase = np.angle(data_ax[:,1,:] * np.conjugate(data_ax[:,2,:]))
dif_phase = np.mean(dif_phase,axis=0)

noise_std = np.std(data_ax, axis=0)
snr = np.abs(data_ax[0]) / (noise_std)
print('SNR:', np.max(snr), np.min(snr))
snr_threshold = 2
mask = snr >= snr_threshold

data_masked = data_avg.copy()
data_masked[~mask] = 0
data_ax_masked = data_ax.copy()
data_ax_masked[:, ~mask] = 0

## B1 calibration
n = 5
s1 = np.abs(data_masked[0])
s2 = np.abs(data_masked[1])
valid = (s1 > 0) & (np.abs(n * s1 - s2) > 1e-12)
theta_deg = np.zeros_like(s1, dtype=float)
r = s2[valid] / s1[valid]
cos_theta = (n * r - 1) / (n - r)
cos_theta = np.clip(cos_theta, -1, 1)
theta_deg[valid] = np.rad2deg(np.arccos(cos_theta))
fa_mean = np.mean(theta_deg[valid])

## Shimming
dif_phase_masked = np.angle(data_ax_masked[:,1,:] * np.conjugate(data_ax_masked[:,2,:]))
dif_phase_masked = np.mean(dif_phase_masked,axis=0)
x_ph = np.arange(len(dif_phase_masked))
y_ph = np.unwrap(dif_phase_masked)
valid = np.isfinite(y_ph) & (y_ph != 0)
slope_ph, intercept_ph = np.polyfit(x_ph[valid], y_ph[valid], 1)
linear_fit_ph = slope_ph * x_ph + intercept_ph

fig, ax = plt.subplots(2, 3, figsize=(10, 8))
fig.suptitle(fig_title)
ax[0,0].plot(np.abs(data_masked[0]))
ax[0,0].set_title('Mag: TR1')
# ax[0,0].set_xlabel("Sample")
ax[0,0].set_ylabel("Magnitude")
ax[0,0].grid(True)

ax[0,1].plot(np.abs(data_masked[1]))
ax[0,1].set_title('Mag: TR2 - Echo 1')
# ax[0,1].set_xlabel("Sample")
ax[0,1].set_ylabel("Magnitude")
ax[0,1].grid(True)

ax[0,2].plot(np.abs(np.flip(data_masked[2])))
ax[0,2].set_title('Mag: TR2 - Echo 2')
# ax[0,2].set_xlabel("Sample")
ax[0,2].set_ylabel("Magnitude")
ax[0,2].grid(True)

ax[1,0].axis("off")

ax[1,1].plot(theta_deg, 'o', label=f"FA mean [deg] = {fa_mean:.2f}")
ax[1,1].set_title('Flip angle [deg]: from E1 TR2/TR1')
# ax[1,1].set_xlabel("Sample")
ax[1,1].set_ylabel("Flip angle [deg]")
ax[1,1].grid(True)
ax[1,1].legend()

ax[1,2].plot(x_ph, y_ph, label="Exp")
ax[1,2].plot(x_ph, linear_fit_ph, "--", label=f"Fit: y = {slope_ph:.4f}x + {intercept_ph:.4f}")
ax[1,2].set_title('Phase diff: TR2 E2-E1')
# ax[1,2].set_xlabel("Sample")
ax[1,2].set_ylabel("Phase")
ax[1,2].grid(True)
ax[1,2].legend()

plt.show()

# fig, ax = plt.subplots(1, 4, figsize=(15, 4))
# fig.suptitle(fig_title)
# ax[0].plot(np.abs(data_masked[0]))
# ax[0].set_title('Mag: TR1')
# ax[0].set_xlabel("Sample")
# ax[0].set_ylabel("Magnitude")
# ax[0].grid(True)

# ax[1].plot(np.abs(data_masked[1]))
# ax[1].set_title('Mag: TR2 - Echo 1')
# ax[1].set_xlabel("Sample")
# ax[1].set_ylabel("Magnitude")
# ax[1].grid(True)

# ax[2].plot(np.abs(np.flip(data_masked[2])))
# ax[2].set_title('Mag: TR2 - Echo 2')
# ax[2].set_xlabel("Sample")
# ax[2].set_ylabel("Magnitude")
# ax[2].grid(True)

# ax[3].plot(np.unwrap(dif_phase_masked))
# ax[3].set_title('Phase diff: TR2')
# ax[3].set_xlabel("Sample")
# ax[3].set_ylabel("Phase")
# ax[3].grid(True)

# plt.show()

# fig, ax = plt.subplots(1, 4, figsize=(15, 4))
# ax[0].plot(np.abs(data_avg[0]))
# ax[0].set_title('Mag: TR1')
# ax[0].set_xlabel("Sample")
# ax[0].set_ylabel("Magnitude")
# ax[0].grid(True)

# ax[1].plot(np.abs(data_avg[1]))
# ax[1].set_title('Mag: TR2 - Echo 1')
# ax[1].set_xlabel("Sample")
# ax[1].set_ylabel("Magnitude")
# ax[1].grid(True)

# ax[2].plot(np.abs(np.flip(data_avg[2])))
# ax[2].set_title('Mag: TR2 - Echo 2')
# ax[2].set_xlabel("Sample")
# ax[2].set_ylabel("Magnitude")
# ax[2].grid(True)

# ax[3].plot(np.unwrap(dif_phase))
# ax[3].set_title('Phase diff: TR2')
# ax[3].set_xlabel("Sample")
# ax[3].set_ylabel("Phase")
# ax[3].grid(True)

# plt.show()
