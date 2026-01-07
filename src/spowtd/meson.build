# Python sources
python_sources = [
  '__init__.py',
  'classify.py',
  'exceptions.py',
  'fit_offsets.py',
  'fpspline.py',
  'load.py',
  'pestfiles.py',
  'plot_recession.py',
  'plot_rise.py',
  'plot_specific_yield.py',
  'plot_time_series.py',
  'plot_transmissivity.py',
  'recession.py',
  'regrid.py',
  'rise.py',
  'schema.sql',
  'set_curvature.py',
  'simulate_recession.py',
  'simulate_rise.py',
  'specific_yield.py',
  'transmissivity.py',
  'user_interface.py',
  'zeta_grid.py',
  'VERSION.txt',
]
py.install_sources(python_sources, subdir: 'spowtd')

# Python test code
test_sources = [
  'test/__init__.py',
  'test/conftest.py',
  'test/markers.py',
  'test/memory.py',
  'test/peatclsm_hydraulic_functions.R',
  'test/test_classify.py',
  'test/test_cli.py',
  'test/test_fit_offsets.py',
  'test/test_load.py',
  'test/test_pestfiles.py',
  'test/test_recession.py',
  'test/test_rise.py',
  'test/test_set_curvature.py',
  'test/test_simulate_recession.py',
  'test/test_simulate_rise.py',
  'test/test_specific_yield.py',
  'test/test_spline.py',
  'test/test_transmissivity.py',
  'test/test_version.py',
]
# Test suite and sample data
py.install_sources(test_sources, subdir: 'spowtd/test')
install_subdir('test/sample_data',
  install_dir: py.get_install_dir() / 'spowtd/test',
  strip_directory: false
)

# Cython extension modules
# 1. Specify dependencies
py_dep = py.dependency()

blas_dep = dependency('flexiblas', required: false)
if not blas_dep.found()
  blas_dep = dependency('openblas', required: false)
endif
if not blas_dep.found()
  blas_dep = dependency('blas', required: true)
endif

gsl_dep = dependency('gsl')

scalar_function_ext = py.extension_module(
  '_scalar_function',
  '_scalar_function.pyx',
  dependencies: [py_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd'
)
spline_ext = py.extension_module(
  '_spline',
  '_spline.pyx',
  dependencies: [py_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd'
)
exts = [spline_ext, scalar_function_ext]

# Testing
pylint = find_program('pylint', required: false)
if not pylint.found()
  pylint = find_program('pylint-3', required: false)
endif

pytest = find_program('pytest', required: false)
if not pytest.found()
  pytest = find_program('pytest-3', required: false)
endif

if pylint.found()
  test('pylint',
    pylint,
    args: [
      '-E',
      '--output-format=colorized',
      '--rcfile=' + meson.project_source_root() / 'pylintrc',
      meson.project_source_root() / 'src'
    ],
    priority: 100,
    suite: 'lint'
  )
else
  message('Pylint not found, skipping')
endif

if pytest.found()
  test('python-tests',
    pytest,
    args: ['--color=yes', meson.project_source_root() / 'src'],
    depends: exts
  )
else
  message('Pytest not found, skipping')
endif
