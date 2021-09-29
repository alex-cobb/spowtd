"""Configure py.test to ignore flymake files

"""

import os


collect_ignore = []  # pylint: disable=invalid-name

for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
    collect_ignore += [
        os.path.join(dirpath, filename.replace('.py', '_flymake.py'))
        for filename in filenames if filename.endswith('.py')]
