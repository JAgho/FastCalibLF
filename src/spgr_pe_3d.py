
import copy
import numpy as np
import pypulseq as pp


def make_double_echo_spgr_pe(
    fov_readout=220e-3,
    fov_phase=(220e-3, 220e-3),
    n_readout=40,
    n_phase=(8, 8),
    pe_coordinates=None,
    readout_axis="z",
    phase_encoding_axes=("x", "y"),
    n_averages=1,
    n_dummy=8,
    flip_angle_deg=15.0,
    rf_duration=200e-6,
    te_1=6e-3,
    te_2=14e-3,
    tr=30e-3,
    readout_time=4e-3,
    prephasing_time=1e-3,
    flyback_time=1e-3,
    rewinding_time=1e-3,
    spoiling_time=2e-3,
    spoiler_cycles=10,
    spoiler_extent=(220e-3, 220e-3, 220e-3),
    rf_spoiling_increment_deg=0.0,
    filename="double_echo_spgr_pe.seq",
):
    """
    Double-echo, same-polarity 3D SPGR.

    pe_coordinates is an (N, 2) array of normalized (kPE1, kPE2)
    coordinates in [-0.5, 0.5]. Its row order is the acquisition order.
    """
    system = pp.Opts(
        max_grad=20,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=10e-6,
        adc_dead_time=00e-6,
    )
    seq = pp.Sequence(system)
    raster = seq.grad_raster_time

    def ceil_raster(t):
        return np.ceil(max(0.0, t) / raster) * raster

    n_phase = np.asarray(n_phase, dtype=int)
    fov_phase = np.broadcast_to(np.asarray(fov_phase, dtype=float), 2)
    if n_phase.shape != (2,) or np.any(n_phase < 1):
        raise ValueError("n_phase must be a pair of positive integers.")

    if pe_coordinates is None:
        p1 = np.arange(n_phase[0]) / n_phase[0] - 0.5
        p2 = np.arange(n_phase[1]) / n_phase[1] - 0.5
        g1, g2 = np.meshgrid(p1, p2, indexing="ij")
        pe_coordinates = np.column_stack((g1.ravel(), g2.ravel()))

    pe_coordinates = np.asarray(pe_coordinates, dtype=float)
    if pe_coordinates.ndim != 2 or pe_coordinates.shape[1] != 2:
        raise ValueError("pe_coordinates must have shape (N, 2).")
    if np.any((pe_coordinates < -0.5) | (pe_coordinates > 0.5)):
        raise ValueError("All PE coordinates must lie in [-0.5, 0.5].")

    readout_axis = readout_axis.lower()
    phase_encoding_axes = tuple(a.lower() for a in phase_encoding_axes)
    if len(phase_encoding_axes) != 2 or len({readout_axis, *phase_encoding_axes}) != 3:
        raise ValueError("readout_axis and the two PE axes must be x, y, z in some order.")

    rf_base = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )

    extent_x, extent_y, extent_z = spoiler_extent
    gx_spoil = pp.make_trapezoid(
        channel="x", area=spoiler_cycles / extent_x,
        duration=spoiling_time, system=system,
    )
    gy_spoil = pp.make_trapezoid(
        channel="y", area=spoiler_cycles / extent_y,
        duration=spoiling_time, system=system,
    )
    gz_spoil = pp.make_trapezoid(
        channel="z", area=spoiler_cycles / extent_z,
        duration=spoiling_time, system=system,
    )
    spoiler_duration = pp.calc_duration(gx_spoil, gy_spoil, gz_spoil)

    delta_k_ro = 1.0 / fov_readout
    tr_index = 0
    actual_te_1 = None
    actual_te_2 = None
    actual_tr = None

    def add_tr(pe_coordinate, acquire=True):
        nonlocal tr_index, actual_te_1, actual_te_2, actual_tr

        rf_phase = np.deg2rad(
            (0.5 * tr_index * (tr_index + 1) * rf_spoiling_increment_deg) % 360.0
        )
        tr_index += 1

        rf = copy.deepcopy(rf_base)
        rf.phase_offset = rf_phase

        g_ro = pp.make_trapezoid(
            channel=readout_axis,
            flat_area=n_readout * delta_k_ro,
            flat_time=readout_time,
            system=system,
        )
        adc_1 = pp.make_adc(
            num_samples=n_readout,
            duration=g_ro.flat_time,
            delay=g_ro.rise_time,
            phase_offset=rf_phase,
            system=system,
        )
        adc_2 = copy.deepcopy(adc_1)

        # Start both echoes at -kmax/2; PE moment remains fixed through both echoes.
        g_ro_pre = pp.make_trapezoid(
            channel=readout_axis,
            area=-g_ro.area / 2,
            duration=prephasing_time,
            system=system,
        )
        pe_area = pe_coordinate * n_phase / fov_phase
        g_pe1 = pp.make_trapezoid(
            channel=phase_encoding_axes[0], area=pe_area[0],
            duration=prephasing_time, system=system,
        )
        g_pe2 = pp.make_trapezoid(
            channel=phase_encoding_axes[1], area=pe_area[1],
            duration=prephasing_time, system=system,
        )

        # Return from +kmax/2 to -kmax/2 for a same-polarity second echo.
        g_flyback = pp.make_trapezoid(
            channel=readout_axis,
            area=-g_ro.area,
            duration=flyback_time,
            system=system,
        )

        # Return to k=0 after echo 2.
        g_ro_rewind = pp.make_trapezoid(
            channel=readout_axis,
            area=-g_ro.area / 2,
            duration=rewinding_time,
            system=system,
        )
        g_pe1_rewind = pp.make_trapezoid(
            channel=phase_encoding_axes[0], area=-pe_area[0],
            duration=rewinding_time, system=system,
        )
        g_pe2_rewind = pp.make_trapezoid(
            channel=phase_encoding_axes[1], area=-pe_area[1],
            duration=rewinding_time, system=system,
        )

        pre_duration = pp.calc_duration(g_ro_pre, g_pe1, g_pe2)
        flyback_duration = pp.calc_duration(g_flyback)
        rewind_duration = pp.calc_duration(g_ro_rewind, g_pe1_rewind, g_pe2_rewind)
        readout_duration = pp.calc_duration(g_ro)

        rf_center = pp.calc_rf_center(rf)[0] + rf.delay
        after_rf_center = pp.calc_duration(rf) - rf_center

        min_te_1 = (
            after_rf_center
            + pre_duration
            + g_ro.rise_time
            + g_ro.flat_time / 2
        )
        if te_1 < min_te_1 - 1e-12:
            raise ValueError(
                f"TE1 too short; "
                f"minimum is {min_te_1 * 1e3:.3f} ms."
            )
        delay_te_1 = ceil_raster(te_1 - min_te_1)
        this_te_1 = min_te_1 + delay_te_1

        min_te_2 = this_te_1 + readout_duration + flyback_duration
        if te_2 < min_te_2 - 1e-12:
            raise ValueError(
                f"TE2 too short; "
                f"minimum is {min_te_2 * 1e3:.3f} ms."
            )
        delay_te_2 = ceil_raster(te_2 - min_te_2)
        this_te_2 = min_te_2 + delay_te_2

        used = (
            pp.calc_duration(rf)
            + pre_duration
            + delay_te_1
            + readout_duration
            + flyback_duration
            + delay_te_2
            + readout_duration
            + rewind_duration
            + spoiler_duration
        )
        if used > tr + 1e-12:
            raise ValueError(
                f"TR too short; "
                f"minimum is {used * 1e3:.3f} ms."
            )
        delay_tr = ceil_raster(tr - used)
        this_tr = used + delay_tr

        seq.add_block(rf)
        seq.add_block(g_ro_pre, g_pe1, g_pe2)
        if delay_te_1 > 0:
            seq.add_block(pp.make_delay(delay_te_1))

        if acquire:
            seq.add_block(g_ro, adc_1)
        else:
            seq.add_block(g_ro)

        seq.add_block(g_flyback)
        if delay_te_2 > 0:
            seq.add_block(pp.make_delay(delay_te_2))

        if acquire:
            seq.add_block(g_ro, adc_2)
        else:
            seq.add_block(g_ro)

        seq.add_block(g_ro_rewind, g_pe1_rewind, g_pe2_rewind)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)
        if delay_tr > 0:
            seq.add_block(pp.make_delay(delay_tr))

        actual_te_1, actual_te_2, actual_tr = this_te_1, this_te_2, this_tr

    # Full dummy TRs at kPE=(0,0), preserving RF-spoiling history.
    for _ in range(n_dummy):
        add_tr(np.zeros(2), acquire=False)

    for _ in range(n_averages):
        for pe_coordinate in pe_coordinates:
            add_tr(pe_coordinate, acquire=True)

    seq.set_definition("Name", "double_echo_spgr_pe")
    seq.set_definition("ReadoutAxis", readout_axis)
    seq.set_definition("PhaseEncodingAxes", list(phase_encoding_axes))
    seq.set_definition("ReadoutFOV", fov_readout)
    seq.set_definition("PhaseFOV", fov_phase.tolist())
    seq.set_definition("ReadoutSamples", n_readout)
    seq.set_definition("PhaseEncodingMatrix", n_phase.tolist())
    seq.set_definition("PhaseEncodingCoordinates", pe_coordinates.tolist())
    seq.set_definition("FlipAngleDeg", flip_angle_deg)
    seq.set_definition("TE1", float(actual_te_1))
    seq.set_definition("TE2", float(actual_te_2))
    seq.set_definition("TR", float(actual_tr))
    seq.set_definition("DummyPulses", n_dummy)
    seq.set_definition("Averages", n_averages)

    timing_ok, timing_errors = seq.check_timing()
    if timing_ok:
        print(
            f"Timing passed: TE1={actual_te_1*1e3:.3f} ms, "
            f"TE2={actual_te_2*1e3:.3f} ms, TR={actual_tr*1e3:.3f} ms"
        )
    else:
        print("Timing errors:")
        for error in timing_errors:
            print(error)

    seq.write(filename)
    return seq


if __name__ == "__main__":
    seq = make_double_echo_spgr_pe(
        n_readout=40, n_phase=(8, 8),
        readout_axis="z", phase_encoding_axes=("x", "y"),
        n_averages=4, n_dummy=20, flip_angle_deg=25,
        te_1=6e-3, te_2=9e-3, tr=30e-3, readout_time=1e-3,
        spoiler_cycles=160, filename="double_echo_spgr_3d.seq",
        )
