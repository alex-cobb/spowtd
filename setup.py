#!/usr/bin/python3

"""Installation script for spowtd

"""

from distutils.core import setup
import glob
import os.path
import re
import subprocess


SCRIPTS = glob.glob('bin/*[!~]')
with open('README.txt') as readme_file:
    LONG_DESCRIPTION = readme_file.read()
    del readme_file


def build_man_pages():
    """Build manual pages for package

    """
    man_sources = glob.glob('doc/*.1.txt')
    for source in man_sources:
        subprocess.check_call(['a2x', '-f', 'manpage', source])
        del source
    return [name[:-4] for name in man_sources]


def get_version():
    """Get project version

    """
    version_file_path = os.path.join(os.path.dirname(__file__),
                                     'VERSION.txt')
    with open(version_file_path) as version_file:
        version_string = version_file.read().strip()
    version_string_re = re.compile('[0-9.]+')
    match = version_string_re.match(version_string)
    if match is None:
        raise ValueError(
            'version string "{}" does not match regexp "{}"'
            .format(version_string, version_string_re.pattern))
    return match.group(0)


setup(name='spowtd',
      version=get_version(),
      description='Scalar parameterization of water table dynamics',
      author='Alex Cobb',
      author_email='alex.cobb@smart.mit.edu',
      long_description=LONG_DESCRIPTION,
      packages=['spowtd',
                'spowtd/test'],
      data_files=[('/usr/share/man/man1', build_man_pages())],
      scripts=SCRIPTS)
