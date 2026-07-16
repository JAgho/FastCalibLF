# %%
import runpy
import numpy as np
import console

from console.service.acquisition_manager import AcquisitionControlManager
from params import ScanParams
from read_data import read_data
from make_magnitude import make_magnitude
from make_phase_ramps import make_phase_ramps
from spgr import SPGR

def main():
    # TBD
    params = {
    "flip_angle": np.pi / 6,
    "rf_duration": 400e-6,
    "tr1": 20e-3,
    "tr2": 100e-3,
    "ro_bandwidth": 20e3,
    "n_samples": 256,
    "fov_projection": 250e-3,
    "delay_rf_to_prephaser": 0.0,
    "delay_prephaser_to_readout": 0.0,
    "delay_readout_to_spoiler": 0.0,
    "double_echo_gap": 0.0,
    "spoiler_area_factor": 4.0,
    "ramp_duration": 200e-6,
    "gradient_correction": 0.0,
    "dummies": 3,
    "averages": 4,
}

    seq = afi_calibration.constructor(**params)

    seq_base = seq


    #nake f0 version that diesnt plot
    runpy.run_path("/home/openimaging/Code/OpenImaging/utilities/calibration/f0_determination.py")

    # with AcquisitionControlManager() as mngr:
    #     mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
    #     acq_data = mngr.acquisition.run(store_unprocessed=False)
    calib_params = ScanParams(console)

    for i in range(3): 

        seq = SPGR(write_seq=True, plot=True)

        with AcquisitionControlManager() as mgnr:
            mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
            acq_data = mngr.acquisition.run()

        raw = acq_data.receive_data
        # TODO: split raw into 
        data = read_data(
            raw, seq_params["readout_length"], seq_params["n_averages"], seq_params["coords"], seq_params["dummies"])
        hybrid = np.fft # TODO

        S1, S2 = make_magnitude(hybrid) # TODO: avg data here
        alpha = fit_alpha(S1, S2)

        hx, hy, hz = make_phase_ramps(hybrid)
        vx, vy, vz = unwrap(hx, hy, hz)
        params_new = make_physical(rx, ry, rz, alpha)
        write_new(calib_params)

    calib_params.print()


if __name__ == "__main__":
    main()
