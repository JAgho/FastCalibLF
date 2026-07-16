import numpy as np
import pypulseq as pp


def make_spoiled_gre_1d(
    fov=220e-3,
    n_readout=128,
    n_dummy = 5,
    n_repetitions=1,
    projection_axis="x",
    flip_angle_deg=15,
    rf_duration=200e-6,
    te=6e-3,
    tr=20e-3,
    readout_time=4e-3,
    prephasing_time=1e-3,
    spoiling_time=2e-3,
    spoiler_cycles=10,
    spoiler_extent=(220e-3, 220e-3, 220e-3),
    filename="spoiled_gre_1d.seq",
):


    # ============================================================
    # Grad Axis
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
    # Grads
    # ============================================================
    delta_k = 1 / fov
    resolution = fov / n_readout

    # ============================================================
    # RF square pulse
    # ============================================================
    
    rf = pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=rf_duration,
        system=system,
        use="excitation",
    )

    # ============================================================
    # Readout grad
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
    # Cálculo del TE
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
    # TR
    # ============================================================
    duration_without_tr_delay = (
        pp.calc_duration(rf)
        + pp.calc_duration(g_pre)
        + te_delay
        + pp.calc_duration(g_readout)
        + spoiler_duration
    )

    tr_delay = tr - duration_without_tr_delay

    if tr_delay < 0:
        raise ValueError(
            f"TR demasiado corto. "
            f"TR mínimo aproximado: "
            f"{duration_without_tr_delay * 1e3:.3f} ms."
        )

    tr_delay = (
        np.floor(tr_delay / seq.grad_raster_time)
        * seq.grad_raster_time
    )

    # ============================================================
    # Sequence
    # ============================================================ 
    for _ in range(n_dummy):

        # Rf excitatiion
        seq.add_block(rf)

        # TR
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))

    for _ in range(n_repetitions):
        #### First TR ####
        # Rf excitatiion
        seq.add_block(rf)

        # Prephase
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

        # TR
        if tr_delay > 0:
            seq.add_block(pp.make_delay(tr_delay))


    # ============================================================
    # Metadatos
    # ============================================================
    seq.set_definition("Name", "spoiled_gre_1d")
    seq.set_definition("ProjectionAxis", projection_axis)
    seq.set_definition("ProjectionFOV", fov)
    seq.set_definition("ReadoutSamples", n_readout)
    seq.set_definition("flip_angle_deg", flip_angle_deg)
    seq.set_definition("TE", te)
    seq.set_definition("TR", tr)

    # ============================================================
    # Comprobación
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
    print(f"Duración RF:                {rf_duration * 1e6:.1f} µs")
    print(f"TE:                         {te * 1e3:.3f} ms")
    print(f"TR:                         {tr * 1e3:.3f} ms")
    print(f"Número de repeticiones:     {n_repetitions}")

    seq.write(filename)

    return seq


if __name__ == "__main__":

    tr = 20e-3
    n_repetitions = 1
    n_dummy = 5

    seq = make_spoiled_gre_1d(
        fov=220e-3,
        n_readout=128,
        n_dummy = n_dummy,
        n_repetitions=n_repetitions,

        # Eje de la proyección: "x", "y" o "z"
        projection_axis="z",

        # Parámetros de RF
        flip_angle_deg=15.0,
        rf_duration=200e-6,

        te=6e-3,
        tr=tr,

        readout_time=4e-3,
        prephasing_time=1e-3,

        spoiling_time=2e-3,
        spoiler_cycles=10,
        spoiler_extent=(
            220e-3,
            220e-3,
            220e-3,
        ),

        filename="spoiled_gre_1d.seq",
    )

    # RF, ADC y gradientes en una única ventana
    seq.plot(
        time_range=(0, (n_dummy + n_repetitions) * tr),
        time_disp="ms",
        grad_disp="mT/m",
        stacked=True,
        show_guides=True,
    )