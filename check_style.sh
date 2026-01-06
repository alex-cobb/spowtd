#!/bin/bash

PKG=spowtd
echo "--------------------------- Running pylint ---------------------------"
pylint-3 ${PKG}
echo "--------------------------- Running tests ----------------------------"
pytest-3 --cov=${PKG}
echo "----------- Displaying html coverage results using Firefox -----------"
coverage3 html --omit='/usr/*' && firefox file://`pwd`/htmlcov/index.html
exit
