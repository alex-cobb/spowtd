"""Plot specific yield"""

import matplotlib.pyplot as plt

import numpy as np

import yaml

import spowtd.specific_yield as specific_yield_mod


def dump_specific_yield(
    parameters, water_level_min_cm, water_level_max_cm, n_points, outfile
):
    """Dump specific yield to a file"""
    # XXX Units
    outfile.write('water_level_cm, specific_yield\n')
    for water_level_cm, specific_yield in zip(
        *grid_specific_yield(
            parameters, water_level_min_cm, water_level_max_cm, n_points
        )
    ):
        outfile.write(f'{water_level_cm}, {specific_yield}\n')


def plot_specific_yield(parameters, water_level_min_cm, water_level_max_cm, n_points):
    """Plot specific yield"""
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_ylabel('Water level, cm')
    axes.set_xlabel('Specific yield')

    water_level_cm, specific_yield = grid_specific_yield(
        parameters, water_level_min_cm, water_level_max_cm, n_points
    )
    axes.plot(specific_yield, water_level_cm, 'b-')

    plt.show()
    return 0


def grid_specific_yield(parameters, water_level_min_cm, water_level_max_cm, n_points):
    """Compute specific yield on a grid"""
    sy_parameters = yaml.safe_load(parameters)['specific_yield']
    specific_yield = specific_yield_mod.create_specific_yield_function(sy_parameters)
    water_level_cm = np.linspace(
        water_level_min_cm, water_level_max_cm, n_points, dtype=float
    )
    sy_1 = specific_yield(water_level_cm * 10)
    return (water_level_cm, sy_1)
