import os
import sys

path = str(os.path.abspath(os.path.dirname(__file__)))
if path not in sys.path[:5]:
    sys.path.insert(1, path)
