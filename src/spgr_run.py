# %%
import runpy
import numpy as np
import console
from console.service.acquisition_manager import AcquisitionControlManager
from params import ScanParams
from read_data import read_data
from make_magnitude import make_magnitude
from make_phase_ramps import make_phase_ramps
from spgr_def_v0 import spgr

seq_params = {
    "readout_length": 128,
    "n_averages": 1,
    "coords": (128, 128, 1),
    "dummies": 5
}

def main():
    # TBD

    runpy.run_path("/home/openimaging/Code/Fast_Calib/FastCalibLF/src/new_f0.py")

    calib_params = ScanParams(console)

    for i in range(3): 

        seq = spgr(**seq_params)

        with AcquisitionControlManager() as mngr:
            mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
            acq_data = mngr.acquisition.run(store_unprocessed=False)

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
