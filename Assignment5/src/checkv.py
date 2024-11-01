"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-09-02
Purpose: Checks package versions
"""


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
    import imblearn
    print('imblearn: {}'.format(imblearn.__version__))
    # fmt: on


if __name__ == "__main__":
    check_versions()
