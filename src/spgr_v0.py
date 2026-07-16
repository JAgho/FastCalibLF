import numpy as np
import pypulseq as pp


def make_spoiled_gre_1d(
    fov=220e-3,
    n_readout=128,
    n_dummy=5,
    n_repetitions=1,
    projection_axes=("x", "y", "z"),
    flip_angle_deg=15,
    rf_duration=200e-6,
    te_1=6e-3,
    te_2=14e-3,
    tr_1=20e-3,
    tr_2_factor=5,
    readout_time=4e-3,
    prephasing_time=1e-3,
    spoiling_time=2e-3,
    spoiler_cycles=10,
    spoiler_extent=(220e-3, 220e-3, 220e-3),
    filename="spoiled_gre_1d.seq",
):
    # ============================================================
    # System limits
    # ============================================================
    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=120,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=10e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    # ============================================================
    # Input parameters
    # ============================================================
    tr_2 = tr_2_factor * tr_1
    if isinstance(projection_axes, str):
        projection_axes = (projection_axes,)
    projection_axes = tuple(axis.lower() for axis in projection_axes)
    delta_k = 1 / fov

    # ============================================================
    # Square RF pulse
    # ============================================================
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )

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

    spoiler_duration = pp.calc_duration(gx_spoil, gy_spoil, gz_spoil)

    # ============================================================
    # Dummy pulses
    # ============================================================
    dummy_delay = tr_1 - pp.calc_duration(rf)
    dummy_delay = np.floor(dummy_delay / seq.grad_raster_time) * seq.grad_raster_time

    for _ in range(n_dummy):
        seq.add_block(rf)

        if dummy_delay > 0:
            seq.add_block(pp.make_delay(dummy_delay))

    # ============================================================
    # TR1 + TR2 block
    # ============================================================
    def add_projection_block(projection_axis):

        # Positive readout
        g_readout_positive = pp.make_trapezoid(
            channel=projection_axis,
            flat_area=n_readout * delta_k,
            flat_time=readout_time,
            system=system,
        )

        adc_1 = pp.make_adc(
            num_samples=n_readout,
            duration=g_readout_positive.flat_time,
            delay=g_readout_positive.rise_time,
            system=system,
        )

        # Negative readout
        g_readout_negative = pp.make_trapezoid(
            channel=projection_axis,
            flat_area=-n_readout * delta_k,
            flat_time=readout_time,
            system=system,
        )

        adc_2 = pp.make_adc(
            num_samples=n_readout,
            duration=g_readout_negative.flat_time,
            delay=g_readout_negative.rise_time,
            system=system,
        )

        # Readout prephaser
        g_pre = pp.make_trapezoid(
            channel=projection_axis,
            area=-g_readout_positive.area / 2,
            duration=prephasing_time,
            system=system,
        )

        # TE1 delay
        rf_center = pp.calc_rf_center(rf)[0] + rf.delay
        time_after_rf_center = pp.calc_duration(rf) - rf_center

        minimum_te_1 = (
            time_after_rf_center
            + pp.calc_duration(g_pre)
            + g_readout_positive.rise_time
            + g_readout_positive.flat_time / 2
        )

        te_delay_1 = te_1 - minimum_te_1

        if te_delay_1 < 0:
            raise ValueError(
                f"TE1 is too short for axis '{projection_axis}'. "
                f"Minimum TE1 is approximately {minimum_te_1 * 1e3:.3f} ms."
            )

        te_delay_1 = np.ceil(te_delay_1 / seq.grad_raster_time) * seq.grad_raster_time

        # Delay between TE1 and TE2
        readout_duration = pp.calc_duration(g_readout_positive)
        inter_echo_delay = te_2 - te_1 - readout_duration

        if inter_echo_delay < 0:
            minimum_te_2 = te_1 + readout_duration

            raise ValueError(
                f"TE2 is too short for axis '{projection_axis}'. "
                f"Minimum TE2 is approximately {minimum_te_2 * 1e3:.3f} ms."
            )

        inter_echo_delay = (
            np.ceil(inter_echo_delay / seq.grad_raster_time)
            * seq.grad_raster_time
        )

        # TR1 delay
        acquisition_duration_1 = (
            pp.calc_duration(rf)
            + pp.calc_duration(g_pre)
            + te_delay_1
            + pp.calc_duration(g_readout_positive)
            + spoiler_duration
        )

        tr_delay_1 = tr_1 - acquisition_duration_1

        if tr_delay_1 < 0:
            raise ValueError(
                f"TR1 is too short for axis '{projection_axis}'. "
                f"Minimum TR1 is approximately {acquisition_duration_1 * 1e3:.3f} ms."
            )

        tr_delay_1 = np.floor(tr_delay_1 / seq.grad_raster_time) * seq.grad_raster_time

        # TR2 delay
        acquisition_duration_2 = (
            pp.calc_duration(rf)
            + pp.calc_duration(g_pre)
            + te_delay_1
            + pp.calc_duration(g_readout_positive)
            + inter_echo_delay
            + pp.calc_duration(g_readout_negative)
            + spoiler_duration
        )

        tr_delay_2 = tr_2 - acquisition_duration_2

        if tr_delay_2 < 0:
            raise ValueError(
                f"TR2 is too short for axis '{projection_axis}'. "
                f"Minimum TR2 is approximately {acquisition_duration_2 * 1e3:.3f} ms."
            )

        tr_delay_2 = np.floor(tr_delay_2 / seq.grad_raster_time) * seq.grad_raster_time

        # Short TR: one echo at TE1
        seq.add_block(rf)
        seq.add_block(g_pre)

        if te_delay_1 > 0:
            seq.add_block(pp.make_delay(te_delay_1))

        seq.add_block(g_readout_positive, adc_1)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        if tr_delay_1 > 0:
            seq.add_block(pp.make_delay(tr_delay_1))

        # Long TR: echo at TE1 and second echo at TE2
        seq.add_block(rf)
        seq.add_block(g_pre)

        if te_delay_1 > 0:
            seq.add_block(pp.make_delay(te_delay_1))

        seq.add_block(g_readout_positive, adc_1)

        if inter_echo_delay > 0:
            seq.add_block(pp.make_delay(inter_echo_delay))

        seq.add_block(g_readout_negative, adc_2)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)

        if tr_delay_2 > 0:
            seq.add_block(pp.make_delay(tr_delay_2))

    # ============================================================
    # Create sequence
    # ============================================================
    for projection_axis in projection_axes:
        for _ in range(n_repetitions):
            add_projection_block(projection_axis)

    # ============================================================
    # Metadata
    # ============================================================
    seq.set_definition("Name", "spoiled_gre_1d")
    seq.set_definition("ProjectionAxes", list(projection_axes))
    seq.set_definition("ProjectionFOV", fov)
    seq.set_definition("ReadoutSamples", n_readout)
    seq.set_definition("FlipAngleDeg", flip_angle_deg)
    seq.set_definition("TE1", te_1)
    seq.set_definition("TE2", te_2)
    seq.set_definition("TR1", tr_1)
    seq.set_definition("TR2", tr_2)
    seq.set_definition("TR2Factor", tr_2_factor)
    seq.set_definition("DummyPulses", n_dummy)

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

    seq.write(filename)

    return seq


if __name__ == "__main__":

    tr_1 = 20e-3
    tr_2_factor = 5
    te_1 = 6e-3
    te_2 = 14e-3

    n_repetitions = 1
    n_dummy = 1
    n_readout = 40
    projection_axes = ("x", "y", "z")

    seq = make_spoiled_gre_1d(
        fov=220e-3,
        n_readout=n_readout,
        n_dummy=n_dummy,
        n_repetitions=n_repetitions,
        projection_axes=projection_axes,
        flip_angle_deg=15.0,
        rf_duration=200e-6,
        te_1=te_1,
        te_2=te_2,
        tr_1=tr_1,
        tr_2_factor=tr_2_factor,
        readout_time=4e-3,
        prephasing_time=1e-3,
        spoiling_time=2e-3,
        spoiler_cycles=n_readout * 4,
        spoiler_extent=(220e-3, 220e-3, 220e-3),
        filename="spoiled_gre_1d.seq",
    )

    tr_2 = tr_2_factor * tr_1
    block_duration = tr_1 + tr_2

    total_duration = (n_dummy * tr_1 + len(projection_axes) * n_repetitions * block_duration)

    seq.plot(
        time_range=(0, total_duration),
        time_disp="ms",
        grad_disp="mT/m",
        stacked=True,
        show_guides=True,
    )
