
import numpy as np
import pypulseq as pp


def make_spoiled_gre_1d(
    fov=220e-3,
    n_readout=128,
    n_dummy=5,
    n_repetitions=1,
    projection_axis="x",
    flip_angle_deg=15,
    rf_duration=200e-6,
    te=6e-3,
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
    # TRs
    # ============================================================
    tr_2 = tr_2_factor * tr_1

    # ============================================================
    # Gradient axis
    # ============================================================
    projection_axis = projection_axis.lower()

    if projection_axis not in ("x", "y", "z"):
        raise ValueError(
            "projection_axis debe ser 'x', 'y' o 'z'."
        )

    if len(spoiler_extent) != 3:
        raise ValueError(
            "spoiler_extent debe contener tres valores: (x, y, z)."
        )

    if any(extent <= 0 for extent in spoiler_extent):
        raise ValueError(
            "Todos los valores de spoiler_extent deben ser positivos."
        )

    if tr_2_factor <= 0:
        raise ValueError(
            "tr_2_factor debe ser positivo."
        )

    # ============================================================
    # System
    # ============================================================
    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=120,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    seq = pp.Sequence(system)

    # ============================================================
    # Spatial encoding
    # ============================================================
    delta_k = 1 / fov
    resolution = fov / n_readout

    # ============================================================
    # Square RF pulse
    # ============================================================
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration,
        system=system,
        use="excitation",
    )

    # ============================================================
    # Readout gradient
    # ============================================================
    g_readout = pp.make_trapezoid(
        channel=projection_axis,
        flat_area=n_readout * delta_k,
        flat_time=readout_time,
        system=system,
    )

    adc = pp.make_adc(
        num_samples=n_readout,
        duration=g_readout.flat_time,
        delay=g_readout.rise_time,
        system=system,
    )

    # ============================================================
    # Prephaser
    # ============================================================
    g_pre = pp.make_trapezoid(
        channel=projection_axis,
        area=-g_readout.area / 2,
        duration=prephasing_time,
        system=system,
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

    spoiler_duration = pp.calc_duration(
        gx_spoil,
        gy_spoil,
        gz_spoil,
    )

    # ============================================================
    # TE
    # ============================================================
    rf_center = pp.calc_rf_center(rf)[0] + rf.delay

    time_after_rf_center = (
        pp.calc_duration(rf) - rf_center
    )

    minimum_te = (
        time_after_rf_center
        + pp.calc_duration(g_pre)
        + g_readout.rise_time
        + g_readout.flat_time / 2
    )

    te_delay = te - minimum_te

    if te_delay < 0:
        raise ValueError(
            f"TE demasiado corto. "
            f"TE mínimo aproximado: {minimum_te * 1e3:.3f} ms."
        )

    te_delay = (
        np.ceil(te_delay / seq.grad_raster_time)
        * seq.grad_raster_time
    )

    # ============================================================
    # TR delays
    # ============================================================
    acquisition_duration = (
        pp.calc_duration(rf)
        + pp.calc_duration(g_pre)
        + te_delay
        + pp.calc_duration(g_readout)
        + spoiler_duration
    )

    tr_delay_1 = tr_1 - acquisition_duration
    tr_delay_2 = tr_2 - acquisition_duration

    if tr_delay_1 < 0:
        raise ValueError(
            f"TR1 demasiado corto. "
            f"TR mínimo aproximado: "
            f"{acquisition_duration * 1e3:.3f} ms."
        )

    if tr_delay_2 < 0:
        raise ValueError(
            f"TR2 demasiado corto. "
            f"TR mínimo aproximado: "
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

    # Los dummy pulses solo contienen RF.
    # Su delay debe completar un TR1 entero.
    dummy_delay = tr_1 - pp.calc_duration(rf)

    if dummy_delay < 0:
        raise ValueError(
            "TR1 es menor que la duración del pulso RF."
        )

    dummy_delay = (
        np.floor(dummy_delay / seq.grad_raster_time)
        * seq.grad_raster_time
    )

    # ============================================================
    # Internal function: complete acquisition TR
    # ============================================================
    def add_acquisition(tr_delay):

        # RF excitation
        seq.add_block(rf)

        # Prephaser
        seq.add_block(g_pre)

        # TE delay
        if te_delay > 0:
            seq.add_block(pp.make_delay(te_delay))

        # ADC readout
        seq.add_block(g_readout, adc)

        # Spoilers
        seq.add_block(
            gx_spoil,
            gy_spoil,
            gz_spoil,
        )

        # Complete TR
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

    # ============================================================
    # Sequence
    # ============================================================

    # Dummy excitations with TR1
    for _ in range(n_dummy):

        seq.add_block(rf)

        if dummy_delay > 0:
            seq.add_block(pp.make_delay(dummy_delay))

    # Each repetition contains one acquisition with TR1
    # followed by one acquisition with TR2.
    for _ in range(n_repetitions):

        # First acquisition: short TR
        add_acquisition(tr_delay_1)

        # Second acquisition: long TR
        add_acquisition(tr_delay_2)

    # ============================================================
    # Metadata
    # ============================================================
    seq.set_definition("Name", "spoiled_gre_1d_two_tr")
    seq.set_definition("ProjectionAxis", projection_axis)
    seq.set_definition("ProjectionFOV", fov)
    seq.set_definition("ReadoutSamples", n_readout)
    seq.set_definition("FlipAngleDeg", flip_angle_deg)
    seq.set_definition("TE", te)
    seq.set_definition("TR1", tr_1)
    seq.set_definition("TR2", tr_2)
    seq.set_definition("TR2Factor", tr_2_factor)
    seq.set_definition("DummyPulses", n_dummy)

    # ============================================================
    # Check
    # ============================================================
    timing_ok, timing_errors = seq.check_timing()

    if timing_ok:
        print("Timing correcto.")
    else:
        print("Errores de timing:")

        for error in timing_errors:
            print(error)

    print(f"Eje de proyección:          {projection_axis}")
    print(f"Resolución nominal:         {resolution * 1e3:.3f} mm")
    print(f"Flip angle nominal:         {flip_angle_deg:.3f}°")
    print(f"Duración RF:                {rf_duration * 1e6:.1f} µs")
    print(f"TE:                         {te * 1e3:.3f} ms")
    print(f"TR1:                        {tr_1 * 1e3:.3f} ms")
    print(f"TR2:                        {tr_2 * 1e3:.3f} ms")
    print(f"Dummy pulses:               {n_dummy}")
    print(f"Número de pares TR1/TR2:    {n_repetitions}")
    print(f"Número de adquisiciones:    {2 * n_repetitions}")

    seq.write(filename)

    return seq


if __name__ == "__main__":

    tr_1 = 20e-3
    tr_2_factor = 5
    n_repetitions = 1
    n_dummy = 1
    n_readout = 40
    seq = make_spoiled_gre_1d(
        fov=220e-3,
        n_readout=n_readout,
        n_dummy=n_dummy,
        n_repetitions=n_repetitions,

        projection_axis="x",

        flip_angle_deg=15.0, # ??
        rf_duration=200e-6,

        te=6e-3,

        tr_1=tr_1,
        tr_2_factor=tr_2_factor,

        readout_time=4e-3,
        prephasing_time=1e-3,

        spoiling_time=2e-3,
        spoiler_cycles=n_readout*4,
        spoiler_extent=(
            220e-3,
            220e-3,
            220e-3,
        ),

        filename="spoiled_gre_1d_two_tr.seq",
    )

    tr_2 = tr_2_factor * tr_1

    total_duration = (
        n_dummy * tr_1
        + n_repetitions * (tr_1 + tr_2)
    )

    # RF, ADC y gradientes en una única ventana
    seq.plot(
        time_range=(0, total_duration),
        time_disp="ms",
        grad_disp="mT/m",
        stacked=True,
        show_guides=True,
    )

