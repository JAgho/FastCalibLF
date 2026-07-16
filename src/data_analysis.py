import numpy as np
import matplotlib.pyplot as plt

data = np.load("src/hybrid.npy")
print(data.shape)
data_x = data[2]
print(data_x.shape)
data_avg = np.mean(data_x, axis=0)
# data_avg = data_x[0]
print(data_avg.shape)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(np.abs(data_avg[0]))
ax[0].set_title('TR1')
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Magnitude")
ax[0].grid(True)

ax[1].plot(np.abs(data_avg[1]))
ax[1].set_title('TR2 - Echo 1')
ax[1].set_xlabel("Sample")
ax[1].set_ylabel("Magnitude")
ax[1].grid(True)

ax[2].plot(np.abs(np.flip(data_avg[2])))
ax[2].set_title('TR2 - Echo 2')
ax[2].set_xlabel("Sample")
ax[2].set_ylabel("Magnitude")
ax[2].grid(True)

# plt.show()
dif_phase = np.angle(data_x[:,1,:] * np.conjugate(data_x[:,2,:]))
# dif_phase = np.angle(data_x[:,1,:]) - np.angle(np.conjugate(data_x[:,2,:]))
dif_phase = np.mean(dif_phase,axis=0)
print(dif_phase.shape)
# plt.plot(dif_phase)
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(np.mean(np.angle(data_x[:,1,:]),axis=0))
ax[0].set_title('Phase: TR2 - Echo 1')
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Magnitude")
ax[0].grid(True)

ax[1].plot(np.mean(np.angle(data_x[:,2,:]),axis=0))
ax[1].set_title('Phase: TR2 - Echo 2')
ax[1].set_xlabel("Sample")
ax[1].set_ylabel("Magnitude")
ax[1].grid(True)

ax[2].plot(np.unwrap(dif_phase))
ax[2].set_title('Phase Diff')
ax[2].set_xlabel("Sample")
ax[2].set_ylabel("Magnitude")
ax[2].grid(True)

plt.show()