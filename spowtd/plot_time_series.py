#!/usr/bin/python3

"""Plot water level and precipitation time series

"""

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


def plot_time_series(
        connection,
        show_accents,
        colors,
        accent_width,
        time_zone_name=None):
    """Plot water level and precipitation time series

    """
    cursor = connection.cursor()

    if time_zone_name is None:
        cursor.execute(
            "SELECT source_time_zone FROM time_grid")
        time_zone_name = cursor.fetchone()[0]
    time_zone = pytz.timezone(time_zone_name)

    fig = plt.figure()
    zeta_axes = fig.add_subplot(2, 1, 1)
    zeta_axes.xaxis_date(tz=time_zone)
    zeta_axes.set_ylabel('Water level, cm')
    rain_axes = fig.add_subplot(2, 1, 2, sharex=zeta_axes)
    rain_axes.xaxis_date(tz=time_zone)
    rain_axes.set_ylabel('Rainfall intensity, mm / h')

    cursor.execute("""
    SELECT epoch,
           zeta_mm / 10 AS zeta_cm,
           rainfall_intensity_mm_h,
           is_jump,
           is_mystery_jump,
           is_interstorm
    FROM water_level AS wl
    JOIN rainfall_intensity AS ri
      ON wl.epoch = ri.from_epoch
    JOIN grid_time_flags AS gtf
      ON gtf.start_epoch = wl.epoch
    ORDER BY epoch""")
    (epoch,
     zeta_cm,
     rain_mm_h,
     is_jump,
     is_mystery_jump,
     is_interstorm) = zip(*cursor)
    (zeta_cm,
     rain_mm_h) = (np.array(v) for v in (zeta_cm,
                                         rain_mm_h))
    mpl_time = dates_mod.epoch2num(epoch)

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

    zeta_axes.plot_date(mpl_time, zeta_cm, 'k-')
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

    rain_axes.plot_date(mpl_time, rain_mm_h, 'k-',
                        drawstyle='steps-post')
    for interval in zip(*rain_storm_intervals):
        rain_axes.axvspan(xmin=interval[0],
                          xmax=interval[1],
                          edgecolor='#ffffff00',
                          facecolor=colors['rain_storm'])

    if show_accents:
        (is_jump,
         is_mystery_jump,
         is_interstorm) = (np.array(v).astype(bool)
                           for v in (is_jump,
                                     is_mystery_jump,
                                     is_interstorm))
        jumps = zeta_cm[:]
        jumps[~is_jump] = np.NaN
        zeta_axes.plot_date(mpl_time, jumps,
                            '-',
                            color=colors['jump'],
                            linewidth=accent_width)
        mystery_jumps = zeta_cm[:]
        mystery_jumps[~is_mystery_jump] = np.NaN
        zeta_axes.plot_date(mpl_time, mystery_jumps,
                            '-',
                            color=colors['mystery_jump'],
                            linewidth=accent_width)

        storm_threshold_mm_h = cursor.execute("""
        SELECT storm_rain_threshold_mm_h
        FROM thresholds""").fetchone()[0]
        storm_rain = rain_mm_h[:]
        storm_rain[rain_mm_h < storm_threshold_mm_h] = np.NaN
        rain_axes.plot_date(mpl_time, storm_rain, '-',
                            color=colors['rain'],
                            linewidth=accent_width,
                            drawstyle='steps-post')

    cursor.close()
    plt.show()
    return 0
