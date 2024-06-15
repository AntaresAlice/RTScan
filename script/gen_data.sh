#!/bin/bash

g++ tools/generate_uniform_data_n_col.cpp -o generate_uniform_data_n_col && ./generate_uniform_data_n_col -n 100000000 -c 3 && rm generate_uniform_data_n_col
python tools/generate_zipf_data.py
python tools/generate_normal_data.py
echo "generate datasets in ./data/ :" 
ls data