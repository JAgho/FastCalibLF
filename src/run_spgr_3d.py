import numpy as np
import console
from cortex.spectrometer import Spectrometer
from console.service.acquisition_manager import AcquisitionControlManager
from spgr_pe_3d import make_double_echo_spgr_pe
import matplotlib.pyplot as plt

def main():
    npe1, npe2, nro, navg = 10, 10, 10, 10
    p1 = np.arange(npe1) / npe1 - 0.5
    p2 = np.arange(npe2) / npe2 - 0.5
    k1, k2 = np.meshgrid(p1, p2, indexing="ij")
    pe_schedule = np.column_stack((k1.ravel(), k2.ravel()))

    seq = make_double_echo_spgr_pe(
        fov_readout=220e-3,
        fov_phase=(220e-3, 220e-3),
        n_readout=nro,
        n_phase=(npe1, npe2),
        pe_coordinates=pe_schedule,
        readout_axis="z",
        phase_encoding_axes=("x", "y"),
        n_averages=navg,
        n_dummy=20,
        flip_angle_deg=25,
        rf_duration=120e-6,
        te_1=12e-3,
        te_2=16.5e-3,
        tr=35e-3,
        readout_time=3e-3,
        spoiler_cycles=160,
        filename="double_echo_spgr_3d.seq",
    )
    running = False
    if running:
        with AcquisitionControlManager() as mngr:
            mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
            acq_data = mngr.acquisition.run()

        raw = acq_data.receive_data
        # print(raw[0])
        #profiles = np.stack([np.asarray(item.processed_data[0]) for item in raw])
        #profiles = np.asarray([item.processed_data[0] for item in raw])
        shape = raw[0].processed_data[0].shape
        profiles = np.stack(
            [item.processed_data[0] for item in raw])  # (ADC event, readout)
        # profiles = profiles.reshape(npe1 * npe2 * navg, shape[-1])  # (ADC event, readout)
        print(f"Profiles shape: {profiles.shape}")
    # (ADC event, readout)[np.asarray(item.processed_data[0,:]) for item in raw], axis=0,)    
        expected = navg * npe1 * npe2 * 2
        if profiles.shape != (expected, nro):
            raise RuntimeError(f"Expected {(expected, nro)} profiles, got {profiles.shape}")

        # Acquisition order: average, PE1, PE2, echo, readout.
        data = profiles.reshape(navg, npe1, npe2, 2, nro).mean(axis=0)
        np.save("double_echo_spgr_3d.npy", data)
    else:
        data = np.load("double_echo_spgr_3d.npy")
        print(f"Loaded data shape: {data.shape}")
    # Optional quick Cartesian reconstruction: image shape (PE1, PE2, RO).
    images = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(data, axes=(0, 1, 3)), axes=(0, 1, 3)),
        axes=(0, 1, 3),
    )
    np.save("double_echo_spgr_3d_images.npy", images)
    
    # Create gif of last axis elements
    frames = []
    for i in range(images.shape[-1]):
        fig, ax = plt.subplots()
        ax.imshow(np.angle(images[:,:,0,i] * np.conj(images[:,:,1,i])), cmap="gray")
        ax.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    
    import imageio
    imageio.mimsave("doubleecho.gif", frames, duration=0.1, loop=0)

if __name__ == "__main__":
    main()
