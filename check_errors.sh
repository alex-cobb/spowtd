#!/bin/bash

set -e

PKG=spowtd
NAMES=`find ${PKG} -name '*.py' -not -path "${PKG}/.ropeproject/config.py" | sort`
echo "--------------------------- Running pylint ---------------------------"
pylint-3 -E ${PKG}
echo "--------------------------- Running tests ----------------------------"
pytest-3 -k test_rise.py
exit
