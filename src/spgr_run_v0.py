# %%
import runpy
import numpy as np
import console

from matplotlib import pyplot as plt

from console.service.acquisition_manager import AcquisitionControlManager
from params import ScanParams
from read_data import read_data
from make_magnitude import make_magnitude
from make_phase_ramps import make_phase_ramps
from spgr_def_v0 import spgr
from fft_data import fft_data

def main():
    # TBD
    params = {
    "fov": 220e-3,
    "n_readout": 40,
    "n_dummy": 10,
    "n_repetitions": 10,
    "projection_axes": ("x", "y", "z"),
    "flip_angle_deg": 15,
    "rf_duration": 200e-6,
    "te_1": 6e-3,
    "te_2": 14e-3,
    "tr_1": 20e-3,
    "tr_2_factor": 5,
    "readout_time": 4e-3,
    "prephasing_time": 1e-3,
    "spoiling_time": 2e-3,
    "spoiler_cycles": 160,
    "spoiler_extent": (220e-3, 220e-3, 220e-3),
}

    seq = spgr(**params)

    seq_base = seq


    #make f0 version that doesnt plot
    # runpy.run_path("/home/openimaging/Code/Fast_Calib/FastCalibLF/src/new_f0.py")

    # with AcquisitionControlManager() as mngr:
    #     mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
    #     acq_data = mngr.acquisition.run(store_unprocessed=False)
    # calib_params = ScanParams(console)
    # print(calib_params)
    with AcquisitionControlManager() as mngr:
        mngr.acquisition.set_sequence(
            sequence=seq,
            parameter=console.parameter,
        )
        acq_data = mngr.acquisition.run()

    raw = acq_data.receive_data

    raw_flat = np.asarray([item.processed_data[0] for item in raw])
    print(raw_flat.shape)
    data = read_data(
            raw_flat, params["n_readout"], params["n_repetitions"], 3, params["n_dummy"])
    print(data.shape)
    hybrid = fft_data(data)

    fig, ax = plt.subplots(1, 3)

    # Plot for all three axes
    for i in range(3):
        ax[i].plot(np.mean(np.abs(hybrid[i]), axis=(0, 1)))

    plt.savefig("Fast_Calib/FastCalibLF/src/hybrid.png")
    np.save("Fast_Calib/FastCalibLF/src/raw.npy", raw)
    np.save("Fast_Calib/FastCalibLF/src/hybrid.npy", hybrid)
    # processed_data = raw.processed_data
    print(hybrid.shape)


    # calib_params.print()


if __name__ == "__main__":
    main()
