# %%
import numpy as np
import console

from matplotlib import pyplot as plt
from cortex.spectrometer import Spectrometer
from console.service.acquisition_manager import AcquisitionControlManager
from params import ScanParams
from read_data import read_data
from make_magnitude import make_magnitude_mean
from make_phase_ramps import make_phase_ramps
from spgr_def_v0 import spgr
from fft_data import fft_data
from new_f0 import F0
from fit_phase_ramps import fit_phase_ramp
from mask_projection_by_snr import mask_projection_by_snr, mask_data
from shim_set import set_b1_scaling, set_shim_offsets

def load_data(path):
    return np.load(path, allow_pickle=True)

def plot_stuff(raw, data, hybrid, masked_hybrid):
    fig, ax = plt.subplots(1, 3)

    # Plot for all three axes
    for i in range(3):
        ax[i].plot(np.mean(np.abs(hybrid[i]), axis=(0, 1)))

    plt.savefig("Fast_Calib/FastCalibLF/src/hybrid.png")
    np.save("Fast_Calib/FastCalibLF/src/raw.npy", raw)
    np.save("Fast_Calib/FastCalibLF/src/data.npy", data)
    np.save("Fast_Calib/FastCalibLF/src/hybrid.npy", hybrid)
    np.save("Fast_Calib/FastCalibLF/src/masked_hybrid.npy", masked_hybrid)
    # processed_data = raw.processed_data
    # print(hybrid.shape)

with Spectrometer() as spec:
    device_config = spec.get_device_configuration()

gpa_gain = device_config.tx.gpa_gain
grad_efficiency = device_config.tx.gradient_efficiency

def main():
    # TBD
    params = {
        "fov": 220e-3,
        "n_readout": 40,
        "n_dummy": 20,
        "n_repetitions": 20,
        "projection_axes": ("x", "y", "z"),
        "flip_angle_deg": 60,
        "rf_duration": 200e-6,
        "te_1": 6e-3,
        "te_2": 10e-3,
        "tr_1": 20e-3,
        "tr_2_factor": 5,
        "readout_time": 3e-3,
        "prephasing_time": 1e-3,
        "spoiling_time": 2e-3,
        "spoiler_cycles": 160,
        "spoiler_extent": (220e-3, 220e-3, 220e-3),
    }
    n_calibs = 1

    # f0 = F0()
    # f0.run()

    calib_params = ScanParams(console)
    for i in range(n_calibs):

        seq = spgr(**params)
        with AcquisitionControlManager() as mngr:
            mngr.acquisition.set_sequence(
                sequence=seq,
                parameter=console.parameter,
            )
            acq_data = mngr.acquisition.run()

        raw = acq_data.receive_data
        # raw = load_data("Fast_Calib/FastCalibLF/src/raw.npy")

        raw_flat = np.asarray([item.processed_data[0] for item in raw])
        data = read_data(
                raw_flat, params["n_readout"], params["n_repetitions"], 3, params["n_dummy"])

        hybrid = fft_data(data)
        mag = make_magnitude_mean(hybrid) # TODO: avg data here

        mask_x, mask_y, mask_z = [mask_projection_by_snr(hybrid[i]) for i in range(3)]

        masked_hybrid = mask_data(hybrid, mask_x, mask_y, mask_z)
        plot_stuff(raw, data, hybrid, masked_hybrid)

        # fit_phase_ramp(
        S1, S2, = mag[0], mag[1]
        alpha = fit_alpha(S1, S2)

        #set B1 scaling factor
        set_b1_scaling(nominal_flip_angle=params["flip_angle_deg"], measured_flip_angle=alpha)

        #adjust shims
        hx, hy, hz = make_phase_ramps(hybrid)
        vx, vy, vz = unwrap(hx, hy, hz)
        params_new = make_physical(rx, ry, rz, alpha)

        set_shim_offsets(x_mt=0.0, y_mt=0.0, z_mt=0.0)
        write_new(calib_params)

    calib_params.print()

if __name__ == "__main__":
    main()
