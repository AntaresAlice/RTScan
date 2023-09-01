#!/usr/bin/env python3

import numpy as np
import os
import stat
from multiprocessing import Pool

a = 4
data_num = '1e8'
n = int(float(data_num))
zipf1_data = np.random.zipf(a, (3, n))
zipf1_data = zipf1_data % np.iinfo(np.uint32).max
zipf1_data.astype("uint32").tofile(f"data/zipf_data_{a}_{data_num}_3.dat")
