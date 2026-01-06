#!/bin/bash

set -e

PKG=spowtd
echo "--------------------------- Running pylint ---------------------------"
pylint-3 -E ${PKG}
echo "--------------------------- Running tests ----------------------------"
pytest-3
exit
