# %%
import numpy as np
#all other scanner imports

import console
from console.servixe.acquisition_manager import AcquisitionControlManager

from params import ScanParams
from read_data import read_data
from make_magnitude import make_magnitude
from make_phase_ramps import make_phase_ramps

def main():
    # TBD
    seq_params = {
        "readout_length": 0,
        "n_averages": 0,
        "coords": 0,
        "dummies": 0
    }

    calib_params = ScanParams(console)

    for i in range(3): 

        seq = SPGR(seq_params, calib_params)

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
