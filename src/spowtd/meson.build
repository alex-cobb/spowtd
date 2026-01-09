# Python sources
python_sources = [
  '__init__.py',
  'classify.py',
  'elapsed_time.py',
  'exceptions.py',
  'fit_offsets.py',
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
  'test/utils.py',
  'test/peatclsm_hydraulic_functions.R',
  'test/test_classify.py',
  'test/test_cli.py',
  'test/test_cumulants.py',
  'test/test_fit_offsets.py',
  'test/test_load.py',
  'test/test_peat_growth.py',
  'test/test_pestfiles.py',
  'test/test_recession.py',
  'test/test_rise.py',
  'test/test_set_curvature.py',
  'test/test_simulate_recession.py',
  'test/test_simulate_rise.py',
  'test/test_specific_yield.py',
  'test/test_specific_yield_ext.py',
  'test/test_spline.py',
  'test/test_transmissivity.py',
  'test/test_transmissivity_ext.py',
  'test/test_version.py',
]
# Test suite and sample data
py.install_sources(test_sources, subdir: 'spowtd/test')
install_subdir('test/sample_data',
  install_dir: py.get_install_dir() / 'spowtd/test',
  strip_directory: false
)

# Cython extension modules
# Specify dependencies
py_dep = py.dependency()
cc = meson.get_compiler('c')

numpy_dep = dependency('numpy')

# Find BLAS
# Search for system BLAS
blas_dep = dependency('flexiblas', required: false)
if not blas_dep.found()
  blas_dep = dependency('openblas', required: false)
endif
if not blas_dep.found()
  blas_dep = dependency('blas', required: false)
endif
if not blas_dep.found()
  # Search for scipy-openblas (Windows/macOS)
  message('No system BLAS found, searching for scipy-openblas...')
  check_scipy_blas = run_command(py, [
    '-c', 'import scipy_openblas; print("scipy-openblas found")'
  ], check: false)
  if check_scipy_blas.returncode() == 0
  message('scipy-openblas found, extracting paths...')
    sciblas_paths = run_command(py, [
      '-c',
      'import scipy_openblas; print(scipy_openblas.get_include()); print(scipy_openblas.get_libdir()); print(scipy_openblas.get_library())'
    ], check: true).stdout().strip().split('\n')
    sciblas_inc = sciblas_paths[0]
    sciblas_libdir = sciblas_paths[1]
    sciblas_name = sciblas_paths[2]
    blas_dep = declare_dependency(
      include_directories: include_directories(sciblas_inc),
      dependencies: cc.find_library(sciblas_name, dirs: sciblas_libdir)
    )
  endif
endif

gsl_dep = dependency('gsl', required: false)

scalar_function_ext = py.extension_module(
  '_scalar_function',
  '_scalar_function.pyx',
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd'
)
spline_ext = py.extension_module(
  '_spline',
  '_spline.pyx',
  dependencies: [py_dep, numpy_dep, gsl_dep, blas_dep],
  install: true,
  subdir: 'spowtd'
)
cumulant_ext = py.extension_module(
  '_cumulant',
  '_cumulant.pyx',
  dependencies: [py_dep, numpy_dep],
  install: true,
  subdir: 'spowtd'
)

inc = include_directories('.')  # For cimport access to .pxds in this dir
subdir('functions')

exts = [
  spline_ext,
  scalar_function_ext,
  cumulant_ext,
  specific_yield_ext,
  transmissivity_ext,
  peat_growth_ext
]

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
