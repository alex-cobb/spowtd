"""User interface for Spowtd

"""

import argparse
import logging
import sys

LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]


def main(argv):
    """CLI for Spowtd

    """
    parser = argparse.ArgumentParser(
        description='Scalar parameterization of water table dynamics')
    subparsers = parser.add_subparsers(help='sub-command help',
                                       dest='task')
    args = parser.parse_args(argv)
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
    if args.task is None:
        pass
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
