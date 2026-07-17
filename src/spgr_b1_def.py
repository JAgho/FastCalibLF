# %%
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt
from math import pi
from console.utilities.sequences.system_settings import system as default_system

# current calibration files are in : "/home/openimaging/Code/openimaging-mri/src/cortex/utilities"


# %%
def spgr_b1(
    n_readout=100,
    n_dummy=20,
    n_repetitions=10,
    flip_angle_deg=60.0,
    rf_duration=200e-6,
    fid_deadtime=80e-6,
    tr_1=20e-3,
    tr_2_factor=5,
    readout_time=4e-3,
    spoiling_time=2e-3,
    spoiler_cycles=160,
    spoiler_extent=(220e-3, 220e-3, 220e-3),
) -> pp.Sequence:
    seq = pp.Sequence(default_system)


    # ============================================================
    # Input parameters
    # ============================================================
    tr_2 = tr_2_factor * tr_1

    if fid_deadtime < system.adc_dead_time:
        raise ValueError(
            f"FID dead time must be at least "
            f"{system.adc_dead_time * 1e6:.1f} us."
        )

    # ============================================================
    # Square non-selective RF pulse
    # ============================================================
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )

    rf_total_duration = pp.calc_duration(rf)

    # ============================================================
    # FID acquisition
    # ============================================================
    adc = pp.make_adc(
        num_samples=n_readout,
        duration=readout_time,
        delay=fid_deadtime,
        system=system,
    )

    adc_total_duration = pp.calc_duration(adc)

    # ============================================================
    # Spoilers
    # ============================================================
    extent_x, extent_y, extent_z = spoiler_extent

    gx_spoil = pp.make_trapezoid(
        channel="x",
        area=spoiler_cycles / extent_x,
        duration=spoiling_time,
        system=system,
    )

    gy_spoil = pp.make_trapezoid(
        channel="y",
        area=spoiler_cycles / extent_y,
        duration=spoiling_time,
        system=system,
    )

    gz_spoil = pp.make_trapezoid(
        channel="z",
        area=spoiler_cycles / extent_z,
        duration=spoiling_time,
        system=system,
    )

    spoiler_duration = pp.calc_duration(
        gx_spoil,
        gy_spoil,
        gz_spoil,
    )

    # ============================================================
    # TR delays
    # ============================================================
    acquisition_duration = (
        rf_total_duration
        + adc_total_duration
        + spoiler_duration
    )

    tr_delay_1 = tr_1 - acquisition_duration
    tr_delay_2 = tr_2 - acquisition_duration

    if tr_delay_1 < 0:
        raise ValueError(
            f"TR1 is too short. Minimum TR1 is approximately "
            f"{acquisition_duration * 1e3:.3f} ms."
        )

    if tr_delay_2 < 0:
        raise ValueError(
            f"TR2 is too short. Minimum TR2 is approximately "
            f"{acquisition_duration * 1e3:.3f} ms."
        )

    tr_delay_1 = (
        np.floor(tr_delay_1 / seq.grad_raster_time)
        * seq.grad_raster_time
    )

    tr_delay_2 = (
        np.floor(tr_delay_2 / seq.grad_raster_time)
        * seq.grad_raster_time
    )

    # ============================================================
    # Sequence blocks
    # ============================================================
    def add_fid_block(tr_delay, acquire=True):
        seq.add_block(rf)

        if acquire:
            seq.add_block(adc)
        else:
            seq.add_block(pp.make_delay(adc_total_duration))

        seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

    # ============================================================
    # Dummy TR pairs
    # ============================================================
    for _ in range(n_dummy):
        add_fid_block(tr_delay_1, acquire=False)
        add_fid_block(tr_delay_2, acquire=False)

    # ============================================================
    # Acquisitions
    # ============================================================
    for _ in range(n_repetitions):
        add_fid_block(tr_delay_1)
        add_fid_block(tr_delay_2)

    # ============================================================
    # Metadata
    # ============================================================
    seq.set_definition("Name", "spoiled_fid")
    seq.set_definition("ReadoutSamples", n_readout)
    seq.set_definition("ReadoutTime", readout_time)
    seq.set_definition("FIDDeadTime", fid_deadtime)
    seq.set_definition("FlipAngleDeg", flip_angle_deg)
    seq.set_definition("TR1", tr_1)
    seq.set_definition("TR2", tr_2)
    seq.set_definition("TR2Factor", tr_2_factor)
    seq.set_definition("DummyPairs", n_dummy)
    seq.set_definition("Repetitions", n_repetitions)

    # ============================================================
    # Timing check
    # ============================================================
    timing_ok, timing_errors = seq.check_timing()

    if timing_ok:
        print("Timing check passed.")
    else:
        print("Timing errors:")

        for error in timing_errors:
            print(error)

    return seq