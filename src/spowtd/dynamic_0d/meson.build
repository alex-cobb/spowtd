# Python sources
python_sources = [
  '__init__.py',
  'dispatch.py',
  'dynamic_solve.py',
]
py.install_sources(python_sources, subdir: 'spowtd/dynamic_0d')

state_0d_ext = py.extension_module(
  '_state_0d',
  '_state_0d.pyx',
  cython_args: [
    '-X', 'cdivision=True',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False'
  ],
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd/dynamic_0d'
)

stepper_0d_ext = py.extension_module(
  '_stepper_0d',
  '_stepper_0d.pyx',
  cython_args: [
    '-X', 'cdivision=True',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False'
  ],
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd/dynamic_0d'
)

simulator_0d_ext = py.extension_module(
  '_simulator_0d',
  '_simulator_0d.pyx',
  cython_args: [
    '-X', 'cdivision=True',
    '-X', 'boundscheck=False',
    '-X', 'wraparound=False'
  ],
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd/dynamic_0d'
)
