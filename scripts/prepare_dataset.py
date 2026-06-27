#!/usr/bin/env python3
"""CLI entry point for dataset preparation.

Usage:
    python scripts/prepare_dataset.py --input /raw/images --output /dataset/custom [options]

Run with --help for full option list.
"""

from alphaface.preprocess.prepare_dataset import main

if __name__ == "__main__":
    main()
