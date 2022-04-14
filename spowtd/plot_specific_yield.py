"""Plot specific yield

"""

import numpy as np

import matplotlib.pyplot as plt


def dump_specific_yield(parameters, water_level_min_cm,
                        water_level_max_cm, n_points,
                        outfile):
    """Dump specific yield to a file

    """
    outfile.write('water_level_cm', 'specific_yield\n')
    for water_level_cm, specific_yield in zip(
            *grid_specific_yield(
                parameters,
                water_level_min_cm, water_level_max_cm,
                n_points)):
        outfile.write('{}, {}\n'.format(water_level_cm,
                                        specific_yield))


def plot_specific_yield(parameters, water_level_min_cm,
                        water_level_max_cm, n_points):
    """Plot specific yield

    """
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlabel('Water level, cm')
    axes.set_ylabel('Specific yield')

    water_level_cm, specific_yield = grid_specific_yield(
        parameters, water_level_min_cm, water_level_max_cm,
        n_points)
    axes.plot(specific_yield, water_level_cm, 'b-')

    plt.show()
    return 0


def grid_specific_yield(parameters, water_level_min_cm,
                        water_level_max_cm, n_points):
    """Compute specific yield on a grid

    """
    water_level_cm = np.linspace(water_level_min_cm,
                                 water_level_max_cm,
                                 n_points,
                                 dtype=float)
    specific_yield = NotImplemented
    return (water_level_cm, specific_yield)
