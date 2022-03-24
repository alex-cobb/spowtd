# spowtd: Scalar parameterization of water table dynamics

Spowtd provides a Python module and script to analyze water table time
series in settings where the behavior of the water table is
essentially determined by the current precipitation and
evapotranspiration.

## Docs

For now, see the PDF user guide in Docs.  For usage hints, just type:
```console
$ spowtd
```

## Bugs

 - Currently the script does not check if the previous steps required
   to do a new step have been completed or not.  So, you may get a
   cryptic error message if, for example, you try to plot the
   recession curve if the recession curve has not been assembled yet.


## Build and install

Build:
python3 setup.py build

Test:
./check_errors.sh

Install:
python3 setup.py install


## Build dependencies

Spowtd is built for Python 3.

Use of the spowtd command-line tool or library requires:
 - python3
 - python3-matplotlib
 - python3-numpy
 - python3-pytz

The following packages are required for the build and tests:
 - asciidoc
 - pdflatex
 - python3
 - pylint-3 (python3-pylint)
 - pytest-3 (python3-pytest)
 - python3-scons

For development, you may additionally want:
 - python3-pycodestyle
 - python3-coverage
 - python3-pytest-cov


## Revision history

Version 0.1.0 - 2021-10-08:
 - Initial packaging.


## Contact information

Alex Cobb</br>
Singapore-MIT Alliance for Research and Technology</br>
Center for Environmental Sensing and Modeling</br>
1 CREATE Way, CREATE Tower #09-03</br>
Singapore 138602</br>
Singapore</br>
tel.: +65 6516 6170</br>
fax.: +65 6778 5654

e-mail: alex.cobb@smart.mit.edu