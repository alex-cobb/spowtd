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


## Installation for use

For end-use, install from local source or GitHub with pip:

```console
pip install .
```
Or
```console
pip install git+https://github.com/alex-cobb/spowtd.git
```

The installation will require:
 - python3
 - python3-matplotlib
 - python3-numpy
 - python3-pytz

To test, run
```console
pytest spowtd
```

## License

The source code is [LICENSE-BSD.txt](BSD-licensed), but compiled distributions are
[LICENSE-GPL.txt](GPL-3.0-or-later) due to the GNU Scientific Library (GSL) dependency.

## Revision history

Version 0.13.0 - 2026-01-09:
 - Specific yield and transmissivity extension modules.
   - This will results in different behavior of specific yield splines between knots
     than in previous versions because of the different boundary conditions used for
     cubic splines in `splrep` (not-a-knot; older versions) and in GSL (natural; this
     and later version).  See the user guide for more.

Version 0.12.0 - 2026-01-08:
 - Add GSL spline extension module.
 - Remove extension module prototype.
 - Specify license terms for code vs. distributed wheels (BSD vs GPL 3+).
 - Update cibuildwheel directives.
 - Add dev docs.

Version 0.11.1 - 2026-01-07:
 - Set up cibuildwheel and fix tests for MacOS and Windows.
 - Skip certain tests depending on environment.

Version 0.11.0 - 2026-01-07:
 - Add a prototype Cython function.
 - Switch to src layout.
 - Docs.

Version 0.10.0 - 2026-01-07:
 - Use meson for build.
 - Docs.

Version 0.9.0 - 2026-01-06:
 - Revert outlier removal code in rise curve assembly
   - If this functionality is desired in the future, it is probably best
     to write a fresh implementation controllable from the CLI.
   - The original implementation will remain available in tag v0.8.0.

Version 0.8.0 - 2026-01-06:
 - Apply isolated changes for rise event stretching and outlier detection
   from doi:10.1002/hyp.70209.
   - Note that *16 tests raise an error* in rise curve assembly.
 - Blacken sources.
 - Rewrap docstrings and comments for smaller diffs.

Version 0.7.0 - 2026-01-03:
 - Weighted rise curve fitting.
 - Fix bugs and deprecation warnings created by upstream changes.
 - Fix dist bugs.
 - Fixtures for more efficient tests.
 - Update contact information.
 - Docs.

Version 0.6.0 - 2023-10-27:
 - Preliminary implementation of event weighting in rise curve assembly.
 - Blacken sources (python-black).
 - Refactoring.

Version 0.5.0 - 2022-09-19:
 - Implement stable matching between storms and rises.
 - Update format of PEST files for consistency.
 - Format Python sources with Black.
 - Fix floating-point comparison issue in .pst tests.

Version 0.4.0 - 2022-05-07:
 - Add curvature to data model, CLI and docs.
 - Implement simulation of recession curves.
 - Load evapotranspiration and add to time series plots.
 - Implement generation of template PEST files for calibration.
 - Add a ceiling parameter to PEATCLSM transmissivity.
 - Make PEATCLSM specific yield instantiation and rise curve
   simulation more efficient.

Version 0.3.0 - 2022-04-25:
 - Preliminary implementations of hydraulic functions.
 - Plotting of hydraulic functions.
 - Simulation of rise curve.
 - Documentation on calibration against rise curve using PEST.

Version 0.2.0 - 2022-03-24:
 - Accommodate gaps in water level time series
 - Better axis label in plot_rise
 - Fix inequality error in classification assertion

Version 0.1.0 - 2021-10-08:
 - Initial packaging.


## Contact information

Alex Cobb</br>
Asian School of the Environment</br>
Nanyang Technological University</br>
50 Nanyang Avenue</br>
Singapore 639798</br>
Singapore

e-mail: alexander.cobb@ntu.edu.sg
