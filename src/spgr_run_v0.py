# %%
import numpy as np
import console

from matplotlib import pyplot as plt
from cortex.spectrometer import Spectrometer
from console.service.acquisition_manager import AcquisitionControlManager
from params import ScanParams
from read_data import read_data_spgr_b1
from make_magnitude import make_magnitude_mean
from make_phase_ramps import make_phase_ramps
from spgr_def_v0 import spgr
from spgr_b1_def import spgr_b1
from fft_data import fft_data
from new_f0 import F0
from fit_phase_ramps import fit_phase_ramp
from mask_projection_by_snr import mask_projection_by_snr, mask_data
from shim_set import set_b1_scaling, set_shim_offsets
from fit_phasemap import linear_shim_from_fieldmap
from fit_alpha import fit_alpha
from mask import make_mask

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
        "n_readout": 100,
        "n_dummy": 20,
        "n_repetitions": 10,
        "flip_angle_deg": 60.0,
        "rf_duration": 200e-6,
        "fid_deadtime": 800e-6,
        "tr_1": 20e-3,
        "tr_2_factor": 5,
        "readout_time": 4e-3,
        "spoiling_time": 2e-3,
        "spoiler_cycles": 160,
        "spoiler_extent": (220e-3, 220e-3, 220e-3),
    }
    n_calibs = 1

    calib_params = ScanParams(console)
    for i in range(n_calibs):

        seq = spgr_b1(**params)
        with AcquisitionControlManager() as mngr:
            mngr.acquisition.set_sequence(
                sequence=seq,
                parameter=console.parameter,
            )
            acq_data = mngr.acquisition.run()

        raw = acq_data.receive_data
        data_avg = read_data_spgr_b1(raw, params["n_repetitions"], params["n_readout"])

        theta_deg = fit_alpha(data_avg)
        # set B1 scaling factor
        b1_scaling = set_b1_scaling(nominal_flip_angle=params["flip_angle_deg"], measured_flip_angle=theta_deg)

        # mask image
        mask = make_mask()

        # adjust shims
        shim = linear_shim_from_fieldmap(
            data=img,
            mask=mask,
            voxel_size_m=(params["fov"] / params["n_readout"],) * 3,
            delta_te=params["te_2"] - params["te_1"],
        )
        gradient_offset = set_shim_offsets(shim["x_mt"], shim["y_mt"], shim["z_mt"])
        calib_params.set_params(
            b1_scaling=b1_scaling,
            larmor_frequency=console.parameter.larmor_frequency,
            gx=gradient_offset.x,
            gy=gradient_offset.y,
            gz=gradient_offset.z,
        )
        calib_params.write_params()

        calib_params.print()

if __name__ == "__main__":
    main()


"""
    mask_x, mask_y, mask_z = [mask_projection_by_snr(hybrid[i]) for i in range(3)]

    masked_hybrid = mask_data(hybrid, mask_x, mask_y, mask_z)
        
    plot_stuff(raw, data, hybrid, masked_hybrid)


    S1, S2 = mag[0], mag[1]
        alpha = fit_alpha(S1, S2)
 hx, hy, hz = make_phase_ramps(hybrid)
        vx, vy, vz = unwrap(hx, hy, hz)
        params_new = make_physical(rx, ry, rz, alpha)
"""