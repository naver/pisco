"""Arm E: build eval_data/longdoc_pages_1500 (see longdoc_build.py for the construction). Extra args pass through."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longdoc_build import main

if __name__ == "__main__":
    main(["--kinds", "pages"] + sys.argv[1:])
