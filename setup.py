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


def build_docs():
    """Build docs for package

    """
    subprocess.check_call(['scons', '-f', 'src/SConstruct'])
    return ['doc/spowtd.1', 'doc/user_guide.pdf']


def get_version():
    """Get project version

    """
    version_file_path = os.path.join(
        os.path.dirname(__file__),
        'spowtd',
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
      package_data={'spowtd': ['VERSION.txt', 'schema.sql'],
                    'spowtd.test': ['sample_data/*.txt']},
      data_files=[('/usr/share/man/man1', build_docs())],
      scripts=SCRIPTS)
