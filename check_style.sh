#!/bin/bash

PKG=spowtd
NAMES=`find ${PKG} -name '*.py' -not -path "${PKG}/.ropeproject/*.py" | sort`
echo "--------------------------- Running pylint ---------------------------"
pylint-3 ${PKG}
echo "---------------------------- Running pep8 ----------------------------"
pycodestyle-3 --repeat --statistics --exclude=${PKG}/.ropeproject/config.py ${NAMES}
echo "--------------------------- Running tests ----------------------------"
pytest-3 --cov=${PKG}
echo "----------- Displaying html coverage results using Firefox -----------"
coverage3 html --omit='/usr/*' && firefox file://`pwd`/htmlcov/index.html
exit
