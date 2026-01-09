# Python sources
python_sources = [
  '__init__.py',
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
  dependencies: [py_dep, numpy_dep],
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
  dependencies: [py_dep, numpy_dep],
  install: true,
  subdir: 'spowtd/dynamic_0d'
)
