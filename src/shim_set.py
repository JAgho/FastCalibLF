import console
import numpy as np
from console.interfaces.acquisition_parameter import Dimensions



def set_shim_offsets(x_mt: float, y_mt: float, z_mt: float) -> None:
    """Set the gradient shim offsets given desired values in mT/m.

    Converts mT/m -> mV using the current GPA gain and gradient efficiency,
    then writes the result to the persistent console.parameter.gradient_offset.
    """

    shims_mv = np.divide((x_mt, y_mt, z_mt), np.multiply(gpa_gain, grad_efficiency))

    console.parameter.gradient_offset = Dimensions(x=shims_mv[0], y=shims_mv[1], z=shims_mv[2])


def set_b1_scaling(nominal_flip_angle: float, measured_flip_angle: float) -> None:
    """Update the B1 scaling from an AFI actual-flip-angle measurement.

    Scales the existing console.parameter.b1_scaling by the ratio of measured
    to nominal flip angle, matching the multiplicative-update convention of the
    flip-angle-sweep calibration. nominal and measured must be in the same units.
    """
    factor = measured_flip_angle / nominal_flip_angle
    console.parameter.b1_scaling = factor * console.parameter.b1_scaling