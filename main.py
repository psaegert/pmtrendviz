import os
import sys

sys.path.append(os.path.abspath(__file__))

from src.pmtrendviz import __main__  # NOQA

if __name__ == '__main__':
    __main__.main()
