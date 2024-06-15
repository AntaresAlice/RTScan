#!/usr/bin/env python3

import numpy as np
import os
import stat
from multiprocessing import Pool

data_num = '1e8'
n = int(float(data_num))

# generate normal data
normal_data = np.random.normal(2**31, 1, (3, n)).astype(np.uint32)
normal_data.tofile(f"data/normal_data_{data_num}_3.dat")
