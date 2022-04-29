"""User interface for Spowtd

"""

import argparse
import logging
import os
import sqlite3
import sys

import spowtd.classify as classify_mod
import spowtd.load as load_mod
import spowtd.recession as recession_mod
import spowtd.rise as rise_mod
import spowtd.plot_recession as recession_plot_mod
import spowtd.plot_rise as rise_plot_mod
import spowtd.plot_specific_yield as specific_yield_plot_mod
import spowtd.plot_time_series as time_series_plot_mod
import spowtd.plot_transmissivity as transmissivity_plot_mod
import spowtd.simulate_recession as simulate_recession_mod
import spowtd.simulate_rise as simulate_rise_mod
import spowtd.zeta_grid as zeta_grid_mod


LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]


def main(argv):
    """CLI for Spowtd

    """
    parser, plot_parser = create_parsers()

    args = parser.parse_args(argv)
    if args.version:
        print(get_version())
        parser.exit()
    if args.task is None:
        parser.print_help()
        parser.exit()

    set_up_logging(args.logfile, args.verbosity)

    if args.task == 'load':
        with sqlite3.connect(args.db) as connection:
            load_mod.load_data(
                connection=connection,
                precipitation_data_file=args.precipitation,
                evapotranspiration_data_file=args.evapotranspiration,
                water_level_data_file=args.water_level,
                time_zone_name=args.timezone)
    elif args.task == 'classify':
        with sqlite3.connect(args.db) as connection:
            classify_mod.classify_intervals(
                connection=connection,
                storm_rain_threshold_mm_h=args.storm_rain_threshold_mm_h,
                rising_jump_threshold_mm_h=args.rising_jump_threshold_mm_h)
    elif args.task == 'set-zeta-grid':
        with sqlite3.connect(args.db) as connection:
            zeta_grid_mod.populate_zeta_grid(
                connection=connection,
                grid_interval_mm=args.water_level_step_mm)
    elif args.task == 'recession':
        with sqlite3.connect(args.db) as connection:
            recession_mod.find_recession_offsets(
                connection=connection,
                reference_zeta_mm=args.reference_zeta_mm)
    elif args.task == 'rise':
        with sqlite3.connect(args.db) as connection:
            rise_mod.find_rise_offsets(
                connection=connection,
                reference_zeta_mm=args.reference_zeta_mm)
    elif args.task == 'plot':
        if args.subtask is None:
            plot_parser.print_help()
            plot_parser.exit()
        if args.subtask in (
                'specific-yield', 'conductivity', 'transmissivity'):
            # No db, no connection
            plot(connection=None,
                 args=args)
        else:
            with sqlite3.connect(args.db) as connection:
                plot(connection=connection,
                     args=args)
    elif args.task == 'simulate':
        if args.subtask is None:
            plot_parser.print_help()
            plot_parser.exit()
        with sqlite3.connect(args.db) as connection:
            simulate(connection=connection,
                     args=args)
    else:
        raise AssertionError('Bad task {}'.format(args.task))
    return 0


def create_parsers():
    """Create spowtd command-line parser and subparsers

    """
    parser = argparse.ArgumentParser(
        description='Scalar parameterization of water table dynamics')
    parser.add_argument(
        '--version',
        help='Print version string and exit',
        action='store_true')

    subparsers = parser.add_subparsers(help='sub-command help',
                                       dest='task')
    load_parser = subparsers.add_parser(
        'load',
        help='Load water level, precipitation and evapotranspiration data')
    add_load_args(load_parser)
    add_shared_args(load_parser)
    del load_parser

    classify_parser = subparsers.add_parser(
        'classify',
        help='Classify data into storm and interstorm intervals')
    add_classify_args(classify_parser)
    add_shared_args(classify_parser)
    del classify_parser

    zeta_grid_parser = subparsers.add_parser(
        'set-zeta-grid',
        help='Set up water level grid for master curves')
    add_zeta_grid_args(zeta_grid_parser)
    add_shared_args(zeta_grid_parser)
    del zeta_grid_parser

    recession_parser = subparsers.add_parser(
        'recession',
        help='Assemble recession curve')
    add_recession_args(recession_parser)
    add_shared_args(recession_parser)
    del recession_parser

    rise_parser = subparsers.add_parser(
        'rise',
        help='Assemble rise curve')
    add_rise_args(rise_parser)
    add_shared_args(rise_parser)
    del rise_parser

    plot_parser = subparsers.add_parser(
        'plot',
        help='Plot data')
    add_plot_args(plot_parser)
    add_shared_args(plot_parser)

    simulate_parser = subparsers.add_parser(
        'simulate',
        help='Simulate water level rise and recession')
    add_simulate_args(simulate_parser)
    add_shared_args(simulate_parser)

    return parser, plot_parser


def set_up_logging(logfile, verbosity):
    """Configure logging for spowtd

    """
    loglevel, is_clipped = get_verbosity(verbosity)
    for handler in logging.root.handlers[:]:
        # Per SO post:
        # https://stackoverflow.com/questions/35898160/\
        #  logging-module-not-writing-to-file?rq=1
        # This is opaque to me, but seems to be necessary.
        logging.root.removeHandler(handler)
    logging.basicConfig(stream=logfile,
                        level=loglevel)
    if is_clipped:
        log = logging.getLogger('spowtd.user_interface')
        log.warning('maximum verbosity exceeded, ignoring flag')


def plot(connection, args):
    """Dispatch to plotting scripts

    """
    if args.subtask == 'time-series':
        colors = time_series_plot_mod.DEFAULT_COLORS.copy()
        time_series_plot_mod.plot_time_series(
            connection=connection,
            show_accents=args.flags,
            colors=colors,
            accent_width=args.highlight_weight,
            time_zone_name=args.timezone)
    elif args.subtask == 'recession':
        recession_plot_mod.plot_recession(
            connection=connection,
            parameters=args.parameters,
            curvature_km=args.curvature_km)
    elif args.subtask == 'rise':
        rise_plot_mod.plot_rise(
            connection=connection,
            parameters=args.parameters)
    elif args.subtask == 'specific-yield':
        if args.dump is not None:
            specific_yield_plot_mod.dump_specific_yield(
                parameters=args.parameters,
                water_level_min_cm=args.water_level_min_cm,
                water_level_max_cm=args.water_level_max_cm,
                n_points=args.n_points,
                outfile=args.dump)
        else:
            specific_yield_plot_mod.plot_specific_yield(
                parameters=args.parameters,
                water_level_min_cm=args.water_level_min_cm,
                water_level_max_cm=args.water_level_max_cm,
                n_points=args.n_points)
    elif args.subtask == 'conductivity':
        raise NotImplementedError
    elif args.subtask == 'transmissivity':
        if args.dump is not None:
            transmissivity_plot_mod.dump_transmissivity(
                parameters=args.parameters,
                water_level_min_cm=args.water_level_min_cm,
                water_level_max_cm=args.water_level_max_cm,
                n_points=args.n_points,
                outfile=args.dump)
        else:
            transmissivity_plot_mod.plot_transmissivity(
                parameters=args.parameters,
                water_level_min_cm=args.water_level_min_cm,
                water_level_max_cm=args.water_level_max_cm,
                n_points=args.n_points)
    else:
        raise AssertionError(
            'Bad plot task {}'.format(args.subtask))


def simulate(connection, args):
    """Dispatch to simulation scripts

    """
    if args.subtask == 'rise':
        simulate_rise_mod.simulate_rise(
            connection=connection,
            parameters=args.parameters,
            outfile=args.output,
            observations_only=args.observations)
    elif args.subtask == 'recession':
        simulate_recession_mod.dump_simulated_recession(
            connection=connection,
            parameter_file=args.parameters,
            curvature_km=args.curvature_km,
            outfile=args.output,
            observations_only=args.observations)
    else:
        raise AssertionError(
            'Bad simulate task {}'.format(args.subtask))


def add_shared_args(parser):
    """Add arguments shared across subparsers

    """
    parser.add_argument(
        '-v', '--verbose', dest='verbosity',
        action='count', default=0,
        help='Write more messages about what is being done')
    parser.add_argument(
        '--logfile',
        metavar='FILE',
        type=argparse.FileType('wt'),
        default=sys.stderr,
        help='File to write status messages, default stderr')


def add_load_args(parser):
    """Add arguments for spowtd load parser

    """
    parser.add_argument(
        'db', metavar='DB', help='Spowtd SQLite3 data file')
    parser.add_argument(
        '-p', '--precipitation', help='Precipitation data file',
        type=argparse.FileType('rt', encoding='utf-8-sig'),
        required=True)
    parser.add_argument(
        '-e', '--evapotranspiration', help='Evapotranspiration data file',
        type=argparse.FileType('rt', encoding='utf-8-sig'),
        required=True)
    parser.add_argument(
        '-z', '--water-level', help='Water level data file',
        type=argparse.FileType('rt', encoding='utf-8-sig'),
        required=True)
    parser.add_argument(
        '--timezone', help='Timezone of time in data files',
        required=True)


def add_classify_args(parser):
    """Add arguments for spowtd classify parser

    """
    parser.add_argument(
        'db', metavar='DB', help='Spowtd SQLite3 data file')
    parser.add_argument(
        '-s', '--storm-rain-threshold-mm-h',
        help='Rainfall intensity > threshold considered a storm',
        type=float, required=True)
    parser.add_argument(
        '-j', '--rising-jump-threshold-mm-h',
        help='Rate of water level increase > threshold considered a storm',
        type=float, required=True)


def add_zeta_grid_args(parser):
    """Add arguments to establish water level grid

    """
    parser.add_argument(
        'db', metavar='DB', help='Spowtd SQLite3 data file')
    parser.add_argument(
        '-d', '--water-level-step-mm',
        help='Water level discretization interval',
        type=float, default=1.0)


def add_recession_args(parser):
    """Add arguments for spowtd recession parser

    """
    parser.add_argument(
        'db', metavar='DB', help='Spowtd SQLite3 data file')
    parser.add_argument(
        '-r', '--reference-zeta-mm',
        help='Water level used to determine origin of time axis',
        type=float, default=None)


def add_rise_args(parser):
    """Add arguments for spowtd rise parser

    """
    parser.add_argument(
        'db', metavar='DB', help='Spowtd SQLite3 data file')
    parser.add_argument(
        '-r', '--reference-zeta-mm',
        help='Water level used to determine origin of storage axis',
        type=float, default=None)


def add_plot_args(parser):
    """Add arguments for spowtd rise parser

    """
    plot_subparsers = parser.add_subparsers(
        help='plotting sub-command help',
        dest='subtask')

    specific_yield_plot_parser = plot_subparsers.add_parser(
        'specific-yield',
        help='Plot specific yield')
    conductivity_plot_parser = plot_subparsers.add_parser(
        'conductivity',
        help='Plot conductivity')
    transmissivity_plot_parser = plot_subparsers.add_parser(
        'transmissivity',
        help='Plot transmissivity')
    for subparser in (
            specific_yield_plot_parser,
            conductivity_plot_parser,
            transmissivity_plot_parser):
        subparser.add_argument(
            'parameters', metavar='YAML',
            type=argparse.FileType('rt'),
            help='YAML hydraulic parameters')
        subparser.add_argument(
            'water_level_min_cm', metavar='WATER_LEVEL_MIN_CM',
            type=float,
            help='Lower end of water level range to plot')
        subparser.add_argument(
            'water_level_max_cm', metavar='WATER_LEVEL_MAX_CM',
            type=float,
            help='Upper end of water level range to plot')
        subparser.add_argument(
            '-n', '--n-points', metavar='N', type=int,
            default=100,
            help='Number of points to plot')
        subparser.add_argument(
            '-d', '--dump',
            type=argparse.FileType('wt'),
            help='Do not plot; dump curve to file as delimited text')
        del subparser
    del specific_yield_plot_parser
    del conductivity_plot_parser,
    del transmissivity_plot_parser

    time_series_plot_parser = plot_subparsers.add_parser(
        'time-series',
        help='Plot water level and precipitation time series')
    time_series_plot_parser.add_argument(
        'db', metavar='SQLITE',
        help='Path to SQLite database')
    time_series_plot_parser.add_argument(
        '-f', '--flags', action='store_true',
        help='Highlight time intervals flagged for storm matching')
    time_series_plot_parser.add_argument(
        '-w', '--highlight-weight', type=float, default=3.0,
        help='Highlight line weight')
    time_series_plot_parser.add_argument(
        '--timezone',
        help='Timezone for time axis, default is original timezone')
    del time_series_plot_parser

    recession_plot_parser = plot_subparsers.add_parser(
        'recession',
        help='Plot master recession curve')
    recession_plot_parser.add_argument(
        'db', metavar='SQLITE',
        help='Path to SQLite database')
    recession_plot_parser.add_argument(
        '-p', '--parameters', metavar='YAML',
        type=argparse.FileType('rt'),
        help='YAML hydraulic parameters')
    recession_plot_parser.add_argument(
        '-k', '--curvature-km', metavar='CURVATURE',
        type=float,
        help='Peat surface curvature for recession simulation, km^-1')
    del recession_plot_parser

    rise_plot_parser = plot_subparsers.add_parser(
        'rise',
        help='Plot master rise curve')
    rise_plot_parser.add_argument(
        'db', metavar='SQLITE',
        help='Path to SQLite database')
    rise_plot_parser.add_argument(
        '-p', '--parameters', metavar='YAML',
        type=argparse.FileType('rt'),
        help='YAML hydraulic parameters')
    del rise_plot_parser


def add_simulate_args(parser):
    """Add arguments for spowtd simulate parser

    """
    simulate_subparsers = parser.add_subparsers(
        help='simulate sub-command help',
        dest='subtask')
    rise_parser = simulate_subparsers.add_parser(
        'rise',
        help='Simulate rise curve')
    rise_parser.add_argument(
        'db', metavar='SQLITE',
        help='Path to SQLite database')
    rise_parser.add_argument(
        'parameters', metavar='YAML',
        type=argparse.FileType('rt'),
        help='YAML hydraulic parameters')
    rise_parser.add_argument(
        '-o', '--output', metavar='FILE',
        help='Write output to file, default stdout',
        type=argparse.FileType('wt'),
        default=sys.stdout)
    rise_parser.add_argument(
        '--observations', action='store_true',
        help='Suppress normal output; just write simulated rise')
    del rise_parser
    recession_parser = simulate_subparsers.add_parser(
        'recession',
        help='Simulate recession curve')
    recession_parser.add_argument(
        'db', metavar='SQLITE',
        help='Path to SQLite database')
    recession_parser.add_argument(
        'parameters', metavar='YAML',
        type=argparse.FileType('rt'),
        help='YAML hydraulic parameters')
    recession_parser.add_argument(
        '-k', '--curvature-km', metavar='CURVATURE',
        type=float,
        help='Peat surface curvature for recession simulation, km^-1',
        required=True)
    recession_parser.add_argument(
        '-o', '--output', metavar='FILE',
        help='Write output to file, default stdout',
        type=argparse.FileType('wt'),
        default=sys.stdout)
    recession_parser.add_argument(
        '--observations', action='store_true',
        help='Suppress normal output; just write simulated recession')
    del recession_parser


def get_verbosity(level_index):
    """Get verbosity of logging for an integer level_index

    Higher levels mean more verbose; the levels are:
      0: ERROR
      1: WARNING
      2: INFO
      3: DEBUG

    Returns a logging debug level and a flag for whether the
    level index was higher than the maximum.

    """
    if level_index >= len(LEVELS):
        level = LEVELS[-1]
        is_clipped = True
    else:
        level = LEVELS[level_index]
        is_clipped = False
    return (level, is_clipped)


def get_version():
    """Get project version

    """
    version_file_path = os.path.join(
        os.path.dirname(__file__),
        'VERSION.txt')
    with open(version_file_path) as version_file:
        return version_file.read().strip()
