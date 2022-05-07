#!/usr/bin/python3

"""Plot water level and precipitation time series

"""

from dataclasses import dataclass

import matplotlib.dates as dates_mod
import matplotlib.pyplot as plt

import numpy as np

import pytz


DEFAULT_COLORS = {
    'interstorm': '#ff000022',
    'zeta_storm': '#00ff0022',
    'rain_storm': '#0000ff22',
    'rain': '#ff0000ff',
    'jump': '#0000ffff',
    'mystery_jump': '#ff00ffff'}


@dataclass
class DataInterval:
    """A contiguous interval of water level and rainfall data

    """
    mpl_time: np.ndarray
    zeta_cm: np.ndarray
    rain_mm_h: np.ndarray
    et_mm_h: np.ndarray
    is_jump: list
    is_mystery_jump: list
    is_interstorm: list


def plot_time_series(
        connection,
        show_accents,
        colors,
        accent_width,
        time_zone_name=None,
        plot_evapotranspiration=False):
    """Plot water level and precipitation time series

    """
    cursor = connection.cursor()

    if time_zone_name is None:
        cursor.execute(
            "SELECT source_time_zone FROM time_grid")
        time_zone_name = cursor.fetchone()[0]
    time_zone = pytz.timezone(time_zone_name)

    fig = plt.figure()
    if plot_evapotranspiration:
        zeta_axes = fig.add_subplot(3, 1, 1)
        rain_axes = fig.add_subplot(3, 1, 2, sharex=zeta_axes)
        rain_axes.set_ylabel('Rainfall intensity,\nmm / h')
        et_axes = fig.add_subplot(3, 1, 3, sharex=zeta_axes)
        et_axes.xaxis_date(tz=time_zone)
        et_axes.set_ylabel('Evapotranspiration,\nmm / h')
    else:
        zeta_axes = fig.add_subplot(2, 1, 1)
        rain_axes = fig.add_subplot(2, 1, 2, sharex=zeta_axes)
        rain_axes.set_ylabel('Rainfall intensity, mm / h')
    zeta_axes.xaxis_date(tz=time_zone)
    zeta_axes.set_ylabel('Water level, cm')
    rain_axes.xaxis_date(tz=time_zone)

    cursor.execute("""
    SELECT DISTINCT data_interval
    FROM grid_time
    WHERE data_interval IS NOT NULL
    ORDER BY data_interval""")
    data_interval_labels = [row[0] for row in cursor.fetchall()]
    if not data_interval_labels:
        raise ValueError('No valid data intervals found')
    data_intervals = []
    for label in data_interval_labels:
        cursor.execute("""
        SELECT wl.epoch,
               zeta_mm / 10 AS zeta_cm,
               rainfall_intensity_mm_h,
               COALESCE(evapotranspiration_mm_h, 'NaN'),
               is_jump,
               is_mystery_jump,
               is_interstorm
        FROM grid_time AS gt
        JOIN water_level AS wl
          ON gt.epoch = wl.epoch
          AND gt.data_interval = ?
        JOIN rainfall_intensity AS ri
          ON wl.epoch = ri.from_epoch
        JOIN grid_time_flags AS gtf
          ON gtf.start_epoch = wl.epoch
        LEFT JOIN evapotranspiration
          ON wl.epoch = evapotranspiration.from_epoch
        ORDER BY wl.epoch""", (label,))
        columns = tuple(zip(*cursor))
        data_intervals.append(
            DataInterval(
                mpl_time=dates_mod.epoch2num(columns[0]),
                zeta_cm=np.array(columns[1], dtype=float),
                rain_mm_h=np.array(columns[2], dtype=float),
                et_mm_h=np.array(columns[3], dtype=float),
                is_jump=columns[4],
                is_mystery_jump=columns[5],
                is_interstorm=columns[6]
            )
        )
        del columns

    cursor.execute("""
    SELECT start_epoch,
           thru_epoch
    FROM zeta_interval
    WHERE interval_type = 'interstorm'""")
    zeta_interstorm_intervals = [
        dates_mod.epoch2num(v)
        for v in zip(*cursor)]
    cursor.execute("""
    SELECT start_epoch,
           thru_epoch
    FROM zeta_interval
    WHERE interval_type = 'storm'""")
    zeta_storm_intervals = [
        dates_mod.epoch2num(v)
        for v in zip(*cursor)]
    cursor.execute("""
    SELECT start_epoch,
           thru_epoch
    FROM storm""")
    rain_storm_intervals = [
        dates_mod.epoch2num(v)
        for v in zip(*cursor)]

    for series in data_intervals:
        zeta_axes.plot_date(series.mpl_time,
                            series.zeta_cm,
                            'k-')
        del series

    for interval in zip(*zeta_interstorm_intervals):
        zeta_axes.axvspan(xmin=interval[0],
                          xmax=interval[1],
                          edgecolor='#ffffff00',
                          facecolor=colors['interstorm'])
    for interval in zip(*zeta_storm_intervals):
        zeta_axes.axvspan(xmin=interval[0],
                          xmax=interval[1],
                          edgecolor='#ffffff00',
                          facecolor=colors['zeta_storm'])

    for series in data_intervals:
        rain_axes.plot_date(series.mpl_time,
                            series.rain_mm_h,
                            'k-',
                            drawstyle='steps-post')
        del series
    for interval in zip(*rain_storm_intervals):
        rain_axes.axvspan(xmin=interval[0],
                          xmax=interval[1],
                          edgecolor='#ffffff00',
                          facecolor=colors['rain_storm'])

    if plot_evapotranspiration:
        for series in data_intervals:
            et_axes.plot_date(series.mpl_time,
                              series.et_mm_h,
                              'k-',
                              drawstyle='steps-post')
            del series

    if show_accents:
        for series in data_intervals:
            jumps = mask_from_list(
                series.zeta_cm,
                np.array(series.is_jump).astype(bool))
            zeta_axes.plot_date(series.mpl_time,
                                jumps,
                                '-',
                                color=colors['jump'],
                                linewidth=accent_width)
            mystery_jumps = mask_from_list(
                series.zeta_cm,
                np.array(series.is_mystery_jump).astype(bool))
            zeta_axes.plot_date(series.mpl_time,
                                mystery_jumps,
                                '-',
                                color=colors['mystery_jump'],
                                linewidth=accent_width)

            storm_threshold_mm_h = cursor.execute("""
            SELECT storm_rain_threshold_mm_h
            FROM thresholds""").fetchone()[0]
            storm_rain = mask_from_list(
                series.rain_mm_h,
                series.rain_mm_h >= storm_threshold_mm_h)
            rain_axes.plot_date(series.mpl_time,
                                storm_rain,
                                '-',
                                color=colors['rain'],
                                linewidth=accent_width,
                                drawstyle='steps-post')
            del series

    cursor.close()
    plt.show()
    return 0


def mask_from_list(array_to_mask, mask):
    """Mask a float array by setting value to NaN according to mask

    mask is a boolean array, used to set values in array_to_mask to
    NaN wherever mask_list is False.

    array_to_mask is unaltered; a copy is returned.

    """
    masked = array_to_mask[:]
    masked[~mask] = np.NaN
    return masked
