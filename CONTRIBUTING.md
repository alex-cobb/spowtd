# Contributing to spowtd

Thank you for your interest in contributing!

This project uses a hybrid build system: **Meson** for compilation and **cibuildwheel**
for distribution.

## Development environment

Work on the Cython extensions requires a C compiler and the Meson build system.

### 1. System Dependencies
* **C Compiler:** `gcc` or `clang`
* **Build Tools:** `ninja-build`, `pkg-config`
* **Libraries:** `sundials` and `gsl`

### 2. Python Environment
A virtual environment and a "no-build-isolation" editable install is recommended. This
way, changes to `.py` files have immediate effect, and `.pyx` files are recompiled.

```console
# Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install meson-python meson ninja cython pytest pylint numpy PyYAML

meson setup build

# Install spowtd in editable mode
pip install --no-build-isolation -e .
```

## Working with Cython

When you modify a .pyx or .c file, you must trigger a rebuild with
```console
pip install --no-build-isolation -e .
```

*Annotations*: Use `cython -a spowtd/foo.pyx` to generate an HTML file showing which
lines of code are interacting with the Python interpreter (yellow highlights). Aim for
white lines in performance-critical loops.

*Header files*: If adding a new C function, declare it in a .pxd file.

## Testing
Tests require `pylint` and `pytest`.  You may additionally want `python3-coverage` and
`python3-pytest-cov`.

To test, run:
```console
meson compile -C build
meson test -C build
```

For convenient test-driven development, a sequence like this works well:
```console
pip install --no-build-isolation -e . --quiet && \
meson setup --reconfigure build && \
meson compile -C build && ]
meson test -C build -v --num-processes 1 --no-rebuild --print-errorlogs
```

By default, `meson test` both checks for errors with `pylint -E` and
runs tests with `pytest`.  The linting step can be run alone with
`meson test --suite lint`, and linting can be skipped with
`meson test --no-suite lint`.


## Pull requests

Before submitting a pull request, please ensure:

1. Code is tested: Any bug fix or new functionality is tested.  Check with `pytest-3
   --cov=spowtd` and `coverage3 html`.  Aim for coverage of 80% or better.

2. Tests pass: pytest shows no errors.

3. Linting: `pylint-3 spowtd` shows no errors or new style complaints.

4. Documentation: Any new features include updates to `doc/user_guide.pdf` or the man
   page if necessary.

5. Git Tracking: You have `git add`-ed any new files. Meson will not include untracked
   files in the build.
   
6. Continuous integration: If `cibuildwheel` fails on any platform, check the logs for
   compiler-specific warnings or missing header errors.

## Docs

Rebuilding the docs requires:
 - asciidoc
 - pdflatex
 - python3-scons


## Distribution

To generate the distributable files manually:
```console
python -m build
```

Note that meson-python only includes files tracked by Git.  New content will need to be
added with `git add` to be included in the package.
