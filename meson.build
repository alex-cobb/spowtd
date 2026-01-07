# Meson build for spowtd
project('spowtd', 'c', 'cython',
  license: 'BSD-2-Clause',
  meson_version: '>= 0.64.0',
  version: run_command('cat', 'VERSION.txt', check: true).stdout().strip(),
  default_options : ['warning_level=2'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

subdir('src/spowtd')

install_data('doc/spowtd.1',
  install_dir: doc_dir
)

install_data('doc/user_guide.pdf',
  install_dir: doc_dir
)
