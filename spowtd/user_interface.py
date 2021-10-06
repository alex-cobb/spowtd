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
import spowtd.zeta_grid as zeta_grid_mod


LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]


def main(argv):
    """CLI for Spowtd

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

    args = parser.parse_args(argv)
    if args.version:
        print(get_version())
        sys.exit(0)
    loglevel, is_clipped = get_verbosity(args.verbosity)
    for handler in logging.root.handlers[:]:
        # Per SO post:
        # https://stackoverflow.com/questions/35898160/\
        #  logging-module-not-writing-to-file?rq=1
        # This is opaque to me, but seems to be necessary.
        logging.root.removeHandler(handler)
    logging.basicConfig(stream=args.logfile,
                        level=loglevel)
    if is_clipped:
        log = logging.getLogger('spowtd.user_interface')
        log.warning('maximum verbosity exceeded, ignoring flag')
    if args.task == 'load':
        with sqlite3.connect(args.db) as connection:
            load_mod.load_data(
                connection=connection,
                precipitation_data_file=args.precipitation,
                evapotranspiration_data_file=args.evapotranspiration,
                water_level_data_file=args.water_level)
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
    else:
        raise AssertionError('Bad task {}'.format(args.task))
    return 0


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


def add_classify_args(parser):
    """Add arguments for spowtd classify parser

    """
    parser.add_argument(
        'db', metavar='DB', help='Spowtd SQLite3 data file')
    parser.add_argument(
        '-s', '--storm-rain-threshold-mm-h',
        help='Rainfall intensity threshold for storms',
        type=float, required=True)
    parser.add_argument(
        '-j', '--rising-jump-threshold-mm-h',
        help='Threshold rate of increase in water level for storms',
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
