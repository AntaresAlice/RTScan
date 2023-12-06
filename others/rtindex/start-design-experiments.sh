#!/bin/bash
set -e
cd "$(dirname "$0")"
cd experiments
# rm -rf build
# mkdir -p build
cd build
# cmake ../src
/usr/bin/env python3 ../src/benchmark.py
# mkdir -p ../../results/local-gpu/experiments
# cp *.csv ../../results/local-gpu/experiments
