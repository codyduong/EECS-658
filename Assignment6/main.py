"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-11-14
Purpose: Compare Models (k-means, GMM, SOM)
"""


import sys
import time
from src.checkv import *
from src.Assignment6 import run
import matplotlib.pyplot as plt

if __name__ == '__main__':
    check_versions()
    run()
    try:
        print("Press Ctrl+C to exit")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        sys.exit(1)
