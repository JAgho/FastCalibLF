# %%
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from math import pi
from console.utilities.sequences.system_settings import system as default_system

# current calibration files are in : "/home/openimaging/Code/openimaging-mri/src/cortex/utilities"


# %%
def spgr(
    flip_angle: float = pi / 6,
    rf_duration: float = 400e-6,
    tr1: float = 20e-3,
    tr2: float = 100e-3,
    ro_bandwidth: float = 20e3,
    n_samples: int = 256,
    fov_projection: float = 250e-3,
    delay_rf_to_prephaser: float = 0.0,
    delay_prephaser_to_readout: float = 0.0,
    delay_readout_to_spoiler: float = 0.0,
    double_echo_gap: float = 0.0,
    spoiler_area_factor: float = 4.0,
    ramp_duration: float = 200e-6,
    gradient_correction: float = 0.0,
    dummies: int = 0,
    averages: int = 1,
) -> pp.Sequence:
    seq = pp.Sequence(default_system)
    seq.set_definition("Name", "spgr_afi_calibration")

    for _ in range(dummies):
        # dummy TR1 (ADC off)
        ...
        # dummy TR2 (ADC off)
        ...

    for grad_channel in ("x", "y", "z"):
        # All gradient block definitions go here since they depend on grad_channel

        for average in range(averages):
            # TR1: spoiled gradient echo, ADC on
            ...

            # TR2: spoiled gradient echo, then double gradient echo
            # (1D projection along grad_channel) before the long TR2 gap, ADC on
            ...

    return seq