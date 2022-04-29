#!/usr/bin/python3

"""Plot master rise curve

"""

import itertools

import matplotlib.pyplot as plt

import numpy as np

import yaml

import spowtd.specific_yield as specific_yield_mod
import spowtd.simulate_rise as simulate_rise_mod


def plot_rise(connection, parameters):
    """Plot master rise curve

    """
    cursor = connection.cursor()

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlabel('Dynamic storage, cm')
    axes.set_ylabel('Water level, cm')

    cursor.execute("""
    SELECT interval_start_epoch AS rise_interval,
           rain_depth_offset_mm / 10 AS storage_cm,
           initial_zeta_mm / 10 AS zeta_cm
    FROM rising_curve_line_segment
    UNION ALL
    SELECT interval_start_epoch AS rise_interval,
           (rain_depth_offset_mm + rain_total_depth_mm) / 10 AS storage_cm,
           final_zeta_mm / 10 AS zeta_cm
    FROM rising_curve_line_segment
    ORDER BY rise_interval, storage_cm""")
    for _, group in itertools.groupby(
            cursor.fetchall(), key=lambda row: row[0]):
        (storage_cm,
         zeta_cm) = zip(*((t, z) for _, t, z in group))
        axes.plot(storage_cm, zeta_cm, '-', color='magenta')

    cursor.execute("""
    SELECT mean_crossing_depth_mm / 10 AS storm_depth_cm,
           zeta_mm / 10 AS zeta_cm
    FROM average_rising_depth
    ORDER BY zeta_mm""")
    (avg_storm_depth_cm,
     avg_zeta_cm) = zip(*cursor)

    axes.plot(avg_storm_depth_cm, avg_zeta_cm, 'b-')

    if parameters is not None:
        plot_simulated_rise(axes, parameters,
                            avg_zeta_cm,
                            np.mean(avg_storm_depth_cm))

    cursor.close()
    plt.show()
    return 0


def plot_simulated_rise(axes, parameters,
                        zeta_grid_cm,
                        mean_storage_cm):
    """Plot simulated rise curve

    """
    sy_parameters = yaml.safe_load(parameters)['specific_yield']
    W_mm = simulate_rise_mod.compute_rise_curve(
        specific_yield=specific_yield_mod.create_specific_yield_function(
            sy_parameters),
        zeta_grid_mm=np.array(zeta_grid_cm, dtype=float) * 10,
        mean_storage_mm=mean_storage_cm * 10)
    axes.plot(W_mm / 10, zeta_grid_cm, 'k--')
