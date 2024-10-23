"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-09-02
Purpose: Checks package versions
"""

def check_versions() -> None:
  # Python version
  import sys
  print('Python: {}'.format(sys.version))
  # scipy
  import scipy
  print('scipy: {}'.format(scipy.__version__))
  # numpy
  import numpy
  print('numpy: {}'.format(numpy.__version__))
  # pandas
  import pandas
  print('pandas: {}'.format(pandas.__version__))
  # scikit-learn
  import sklearn
  print('sklearn: {}'.format(sklearn.__version__))

if __name__ == "__main__":
  check_versions()