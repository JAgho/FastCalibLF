# %%
import runpy
import numpy as np
import console

from console.service.acquisition_manager import AcquisitionControlManager
from params import ScanParams
from read_data import read_data
from make_magnitude import make_magnitude
from make_phase_ramps import make_phase_ramps
from spgr_v0 import spgr

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
    runpy.run_path("/home/openimaging/Code/OpenImaging/utilities/calibration/f0_determination.py")

    # with AcquisitionControlManager() as mngr:
    #     mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
    #     acq_data = mngr.acquisition.run(store_unprocessed=False)
    calib_params = ScanParams(console)
    print(calib_params)
    with AcquisitionControlManager() as mngr:
        mngr.acquisition.set_sequence(
            sequence=seq,
            parameter=console.parameter,
        )
        acq_data = mngr.acquisition.run()

    raw = acq_data.receive_data
    print(raw.size)

    # calib_params.print()


if __name__ == "__main__":
    main()
