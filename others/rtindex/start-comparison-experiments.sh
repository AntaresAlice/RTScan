#!/bin/bash
set -e
cd "$(dirname "$0")"
cd index-prototype
rm -rf build
mkdir -p build
cd build
cmake ../src
make
./index_prototype
mkdir -p ../../results/local-gpu
cp *.csv ../../results/local-gpu
