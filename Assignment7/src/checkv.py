"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-11-14
Purpose: Checks package versions

- S/O | How do I check the versions of Python modules? -> Answer: https://stackoverflow.com/a/32965521/
"""

from importlib.metadata import version


def check_versions() -> None:
    # fmt: off
    import sys
    print('Python: {}'.format(sys.version))
    import scipy
    print('scipy: {}'.format(scipy.__version__))
    import numpy
    print('numpy: {}'.format(numpy.__version__))
    import pandas
    print('pandas: {}'.format(pandas.__version__))
    import sklearn
    print('sklearn: {}'.format(sklearn.__version__))
    print('minisom: {}'.format(version('minisom')))
    print('matplotlib: {}'.format(version('matplotlib')))
    # fmt: on


if __name__ == '__main__':
    check_versions()
