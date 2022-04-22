#!/usr/bin/python3

"""Simulate master rise curve

"""


import numpy as np

import scipy.integrate as integrate_mod

import yaml

import spowtd.specific_yield as specific_yield_mod


def simulate_rise(connection, parameters, outfile,
                  observations_only):
    """Simulate master rise curve

    """
    sy_parameters = yaml.safe_load(parameters)['specific_yield']
    specific_yield = specific_yield_mod.create_specific_yield_function(
        sy_parameters)
    cursor = connection.cursor()
    cursor.execute("""
    SELECT mean_crossing_depth_mm AS dynamic_storage_mm,
           zeta_mm
    FROM average_rising_depth
    ORDER BY zeta_mm""")
    (avg_storage_mm,
     avg_zeta_mm) = (
         np.array(a, dtype=float) for a in zip(*cursor))
    W_mm = compute_rise_curve(
        specific_yield,
        zeta_grid_mm=avg_zeta_mm,
        mean_storage_mm=avg_storage_mm.mean())
    if observations_only:
        outfile.write('# Rise curve simulation vector\n')
        yaml.dump(
            W_mm.tolist(),
            outfile)
    else:
        yaml.dump(
            ([['Water level, mm',
               'Measured storage, mm',
               'Simulated storage, mm']] +
             list(list(item) for item in
                  zip(avg_zeta_mm.tolist(),
                      avg_storage_mm.tolist(),
                      W_mm.tolist()))),
            outfile)


def compute_rise_curve(specific_yield, zeta_grid_mm,
                       mean_storage_mm=0.0):
    """Compute rise curve on a specified grid

    Returns dynamic storage on the given grid.

    If the desired mean storage is given, the rise curve is adjusted
    so its mean matches this value.

    """
    dW_mm = np.empty(zeta_grid_mm.shape, dtype=float)
    dW_mm[0] = 0.0
    i = 1
    for zeta_mm in zeta_grid_mm[1:]:
        dW_mm[i] = integrate_mod.quad(
            specific_yield,
            zeta_grid_mm[i - 1],
            zeta_grid_mm[i])[0]
        i += 1
    W_mm = np.cumsum(dW_mm)
    W_mm += mean_storage_mm - W_mm.mean()
    return W_mm
