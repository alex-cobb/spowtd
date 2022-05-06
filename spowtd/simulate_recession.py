#!/usr/bin/python3

"""Simulate master recession curve

"""

import numpy as np

import scipy.integrate as integrate_mod

import yaml

import spowtd.specific_yield as specific_yield_mod
import spowtd.transmissivity as transmissivity_mod


def dump_simulated_recession(connection, parameter_file, outfile,
                             observations_only):
    """Dump master recession curve to file

    Sequence is from highest to lowest water level.

    """
    (avg_elapsed_time_d,
     avg_zeta_cm,
     elapsed_time_d) = simulate_recession(
         connection, parameter_file)

    if observations_only:
        outfile.write('# Recession curve simulation vector\n')
        yaml.dump(
            list(reversed(elapsed_time_d.tolist())),
            outfile)
    else:
        yaml.dump(
            ([['Water level, mm',
               'Measured elapsed time, d',
               'Simulated elapsed time, d']] +
             list(reversed(
                 list(list(item) for item in
                      zip(avg_zeta_cm.tolist(),
                          avg_elapsed_time_d.tolist(),
                          elapsed_time_d.tolist()))))),
            outfile)


def simulate_recession(connection, parameter_file):
    """Simulate master recession curve

    """
    cursor = connection.cursor()
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM curvature WHERE is_valid)")
    if not cursor.fetchone()[0]:
        raise ValueError(
            'Site curvature must be set to simulate recession')
    cursor.execute(
        "SELECT curvature_m_km2 FROM curvature")
    curvature_m_km2 = cursor.fetchone()[0]
    cursor.execute("""
    SELECT CAST(elapsed_time_s AS double precision)
             / (3600 * 24) AS elapsed_time_d,
           zeta_mm / 10 AS zeta_cm
    FROM average_recession_time
    ORDER BY zeta_mm""")
    (avg_elapsed_time_d,
     avg_zeta_cm) = (np.array(v, dtype=float) for v in zip(*cursor))
    # Mean evapotranspiration in recession intervals
    cursor.execute("""
    SELECT avg(evapotranspiration_mm_h) * 24
             AS evapotranspiration_mm_d
    FROM evapotranspiration AS e
    JOIN recession_interval AS ri
      ON e.from_epoch = ri.start_epoch""")
    et_mm_d = cursor.fetchone()[0]
    assert et_mm_d >= 0, et_mm_d
    # XXX Hack for Congo data, which actually give ET in mm
    # XXX on 3-hour intervals (mm over 3 h)
    et_mm_d /= 3.0
    cursor.close()
    del cursor
    del connection

    parameters = yaml.safe_load(parameter_file)
    # XXX Hack for PEATCLSM parameterization, which gives
    # XXX transmissivity in m2 / s
    if parameters['transmissivity']['type'] == 'peatclsm':
        transmissivity_m2_s = (
            transmissivity_mod.create_transmissivity_function(
                parameters['transmissivity']))

        def transmissivity_m2_d(zeta_mm):
            """Compute transmissivy in m2 / d

            Computed from PEATCLSM transmissivity in m2 / s

            """
            return transmissivity_m2_s(zeta_mm) * 24 * 3600

    else:
        transmissivity_m2_d = (
            transmissivity_mod.create_transmissivity_function(
                parameters['transmissivity']))

    elapsed_time_d = compute_recession_curve(
        specific_yield=(
            specific_yield_mod.create_specific_yield_function(
                parameters['specific_yield'])),
        transmissivity_m2_d=transmissivity_m2_d,
        zeta_grid_mm=np.array(avg_zeta_cm, dtype=float) * 10,
        mean_elapsed_time_d=np.mean(avg_elapsed_time_d),
        curvature_km=curvature_m_km2 * 1e-3,
        et_mm_d=et_mm_d)

    return (avg_elapsed_time_d,
            avg_zeta_cm,
            elapsed_time_d)


def compute_recession_curve(specific_yield, transmissivity_m2_d,
                            zeta_grid_mm, mean_elapsed_time_d,
                            curvature_km, et_mm_d):
    """Compute recession curve on a specified grid

    Returns elapsed time in days on the given grid.

    If the desired mean elapsed time is given, the recession curve is
    adjusted so its mean matches this value.

    """
    assert et_mm_d >= 0
    assert curvature_km >= 0
    dt_d = np.empty(zeta_grid_mm.shape, dtype=float)
    dt_d[0] = 0.0

    def f(zeta_mm):
        """Sy(H_mm) / (-et + Laplacian * Tt(H_mm))

        See Cobb and Harvey (2019) eqn 8 for further explanation.

        """
        return specific_yield(zeta_mm) / (
            -et_mm_d - curvature_km * transmissivity_m2_d(zeta_mm))

    i = 1
    for zeta_mm in zeta_grid_mm[1:]:
        dt_d[i] = integrate_mod.quad(
            f,
            zeta_grid_mm[i - 1],
            zeta_grid_mm[i])[0]
        i += 1
    elapsed_time_d = np.cumsum(dt_d)
    elapsed_time_d += mean_elapsed_time_d - elapsed_time_d.mean()
    return elapsed_time_d
