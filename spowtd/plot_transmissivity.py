"""Plot transmissivity

"""

import matplotlib.pyplot as plt

import numpy as np

import yaml

import spowtd.transmissivity as transmissivity_mod


def dump_transmissivity(parameters, water_level_min_cm,
                        water_level_max_cm, n_points,
                        outfile):
    """Dump specific yield to a file

    """
    # XXX Units
    outfile.write('water_level_cm, transmissivity_m2_s\n')
    for water_level_cm, transmissivity in zip(
            *grid_transmissivity(
                parameters,
                water_level_min_cm, water_level_max_cm,
                n_points)):
        outfile.write('{}, {}\n'.format(water_level_cm,
                                        transmissivity))


def plot_transmissivity(parameters, water_level_min_cm,
                        water_level_max_cm, n_points):
    """Plot specific yield

    """
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_ylabel('Water level, cm')
    axes.set_xlabel('Transmissivity, m^2 / s')
    axes.set_xscale('log')

    water_level_cm, transmissivity_m2_s = grid_transmissivity(
        parameters, water_level_min_cm, water_level_max_cm,
        n_points)
    axes.plot(transmissivity_m2_s, water_level_cm, 'b-')

    plt.show()
    return 0


def grid_transmissivity(parameters, water_level_min_cm,
                        water_level_max_cm, n_points):
    """Compute specific yield on a grid

    """
    T_parameters = yaml.safe_load(parameters)['transmissivity']
    transmissivity = transmissivity_mod.create_transmissivity_function(
        T_parameters)
    water_level_cm = np.linspace(water_level_min_cm,
                                 water_level_max_cm,
                                 n_points,
                                 dtype=float)
    T_m2_s = transmissivity(water_level_cm * 10)
    return (water_level_cm, T_m2_s)
