# Python sources
python_sources = [
  '__init__.py',
  'specific_yield.py',
  'transmissivity.py'
]
py.install_sources(python_sources, subdir: 'spowtd/functions')

specific_yield_ext = py.extension_module(
  '_specific_yield',
  '_specific_yield.pyx',
  cython_args: [
    '-X', 'cdivision=True',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False'
  ],
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  include_directories: inc,  # Access to spline pxd defs with cimport
  install: true,
  subdir: 'spowtd/functions'
)

transmissivity_ext = py.extension_module(
  '_transmissivity',
  '_transmissivity.pyx',
  cython_args: [
    '-X', 'cdivision=True',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False'
  ],
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  include_directories: inc,  # Access to spline pxd defs with cimport
  install: true,
  subdir: 'spowtd/functions'
)
