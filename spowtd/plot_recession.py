#!/usr/bin/python3

"""Plot master recession curve"""

import itertools

import matplotlib.pyplot as plt

import spowtd.simulate_recession as simulate_recession_mod


def plot_recession(connection, parameters):
    """Plot master recession curve"""
    cursor = connection.cursor()

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlabel('Elapsed time, d')
    axes.set_ylabel('Water level, cm')

    cursor.execute(
        """
    SELECT start_epoch AS zeta_interval,
           (epoch - start_epoch + time_offset_s) / (3600. * 24)
             AS elapsed_time_d,
           zeta_mm / 10 AS zeta_cm
    FROM zeta_interval AS zi
    JOIN recession_interval AS ri
      USING (start_epoch, interval_type)
    JOIN water_level AS wl
      ON wl.epoch >= zi.start_epoch
      AND wl.epoch <= zi.thru_epoch
    WHERE interval_type = 'interstorm'
    ORDER BY start_epoch, elapsed_time_d"""
    )
    for _, group in itertools.groupby(cursor.fetchall(), key=lambda row: row[0]):
        (elapsed_time_d, zeta_cm) = zip(*((t, z) for _, t, z in group))
        axes.plot(elapsed_time_d, zeta_cm, '-', color='magenta')

    cursor.execute(
        """
    SELECT CAST(elapsed_time_s AS double precision)
             / (3600 * 24) AS elapsed_time_d,
           zeta_mm / 10 AS zeta_cm
    FROM average_recession_time"""
    )
    (avg_elapsed_time_d, avg_zeta_cm) = zip(*cursor)
    cursor.close()

    axes.plot(avg_elapsed_time_d, avg_zeta_cm)

    if parameters is not None:
        (_, _, elapsed_time_d) = simulate_recession_mod.simulate_recession(
            connection, parameters
        )
        axes.plot(elapsed_time_d, avg_zeta_cm, 'k--')

    plt.show()
    return 0
