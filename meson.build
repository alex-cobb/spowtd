# Meson build for spowtd
project('spowtd', 'c', 'cython',
  license: 'BSD-2-Clause',
  meson_version: '>= 0.64.0',
  version: run_command('cat', 'VERSION.txt', check: true).stdout().strip(),
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Docs
doc_dir = py.get_install_dir() / 'spowtd' / 'doc'

install_data('doc/spowtd.1',
  install_dir: doc_dir
)

install_data('doc/user_guide.pdf',
  install_dir: doc_dir
)

py.extension_module(
    'hello',
    'src/hello.pyx',  # Path to source
    install: true,
    subdir: 'spowtd'  # Destination in package
)

# Python sources
python_sources = [
  'spowtd/__init__.py',
  'spowtd/classify.py',
  'spowtd/fit_offsets.py',
  'spowtd/load.py',
  'spowtd/pestfiles.py',
  'spowtd/plot_recession.py',
  'spowtd/plot_rise.py',
  'spowtd/plot_specific_yield.py',
  'spowtd/plot_time_series.py',
  'spowtd/plot_transmissivity.py',
  'spowtd/recession.py',
  'spowtd/regrid.py',
  'spowtd/rise.py',
  'spowtd/schema.sql',
  'spowtd/set_curvature.py',
  'spowtd/simulate_recession.py',
  'spowtd/simulate_rise.py',
  'spowtd/specific_yield.py',
  'spowtd/spline.py',
  'spowtd/transmissivity.py',
  'spowtd/user_interface.py',
  'spowtd/zeta_grid.py',
  'spowtd/test/__init__.py',
  'spowtd/test/conftest.py',
  'spowtd/test/peatclsm_hydraulic_functions.R',
  'spowtd/test/test_classify.py',
  'spowtd/test/test_cli.py',
  'spowtd/test/test_fit_offsets.py',
  'spowtd/test/test_hello.py',
  'spowtd/test/test_load.py',
  'spowtd/test/test_pestfiles.py',
  'spowtd/test/test_recession.py',
  'spowtd/test/test_rise.py',
  'spowtd/test/test_set_curvature.py',
  'spowtd/test/test_simulate_recession.py',
  'spowtd/test/test_simulate_rise.py',
  'spowtd/test/test_specific_yield.py',
  'spowtd/test/test_transmissivity.py',
  'spowtd/test/test_version.py',
  'spowtd/VERSION.txt',
]
py.install_sources(python_sources, subdir: 'spowtd')

# Test suite and sample data
py.install_sources(['spowtd/test/__init__.py'], subdir: 'spowtd/test')
install_subdir('spowtd/test/sample_data',
  install_dir: py.get_install_dir() / 'spowtd/test',
  strip_directory: false
)

# Testing
pylint = find_program('pylint-3', required: false)
if not pylint.found()
  pylint = find_program('pylint', required: false)
endif

pytest = find_program('pytest-3', required: false)
if not pytest.found()
  pytest = find_program('pytest', required: false)
endif

# Add the source directory to PYTHONPATH so pytest sees the source folder
envdata = environment()
envdata.append('PYTHONPATH', meson.current_source_dir())

if pylint.found()
  test('pylint',
    pylint,
    args: ['-E', 'spowtd'],
    env: envdata,
    priority: 100,
    suite: 'lint'
  )
else
  message('Pylint not found, skipping')
endif

if pytest.found()
  test('python-tests',
    pytest,
    args: [meson.current_source_dir() / 'spowtd/test'],
    env: envdata,
    timeout: 100
  )
else
  message('Pytest not found, skipping')
endif

# Cython Extension Module t.c.
# py.extension_module(
#     'solver_extension',
#     'src/solver.pyx',
#     dependencies: [py_dep],
#     install: true,
#     subdir: 'spowtd'
# )
