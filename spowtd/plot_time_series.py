#!/usr/bin/python3

"""Plot water level and precipitation time series

"""

import argparse
import sqlite3

import matplotlib.dates as dates_mod
import matplotlib.pyplot as plt

import numpy as np


DEFAULT_COLORS = {
    'interstorm': '#ff000022',
    'zeta_storm': '#00ff0022',
    'rain_storm': '#0000ff22',
    'rain': '#ff0000ff',
    'jump': '#0000ffff',
    'mystery_jump': '#ff00ffff'}


def main(argv):
    """CLI to plot water level and precipitation time series

    """
    parser = argparse.ArgumentParser(
        description='Plot water level and precipitation time series')
    parser.add_argument(
        'db', metavar='SQLITE',
        help='Path to SQLite database')
    parser.add_argument(
        '-f', '--flags', action='store_true',
        help='Highlight time intervals flagged for storm matching')
    parser.add_argument(
        '-w', '--highlight-weight', type=float, default=3.0,
        help='Highlight line weight')
    args = parser.parse_args(argv)
    colors = DEFAULT_COLORS.copy()
    with sqlite3.connect(args.db) as connection:
        return plot_time_series(
            connection=connection,
            show_accents=args.flags,
            colors=colors,
            accent_width=args.highlight_weight)


def plot_time_series(
        connection,
        show_accents,
        colors,
        accent_width):
    """Plot water level and precipitation time series

    """
    cursor = connection.cursor()

    fig = plt.figure()
    zeta_axes = fig.add_subplot(2, 1, 1)
    zeta_axes.set_ylabel('Water level, cm')
    rain_axes = fig.add_subplot(2, 1, 2, sharex=zeta_axes)
    rain_axes.set_ylabel('Rainfall intensity, mm / h')

    cursor.execute("""
    SELECT epoch,
           zeta_mm / 10 AS zeta_cm,
           rainfall_intensity_mm_h,
           is_raining,
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
     is_raining,
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
        (is_raining,
         is_jump,
         is_mystery_jump,
         is_interstorm) = (np.array(v).astype(bool)
                           for v in (is_raining,
                                     is_jump,
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

        raining = rain_mm_h[:]
        raining[~is_raining] = np.NaN
        rain_axes.plot_date(mpl_time, rain_mm_h, '-',
                            color=colors['rain'],
                            linewidth=accent_width,
                            drawstyle='steps-post')

    cursor.close()
    plt.show()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
