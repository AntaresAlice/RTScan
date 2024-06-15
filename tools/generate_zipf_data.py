#!/usr/bin/env python3

import numpy as np
import os
import stat
from multiprocessing import Pool

data_num = '1e8'
n = int(float(data_num))

# generate zipf data
for a in [1.1, 1.3, 1.5]:
    zipf_data = np.random.zipf(a, (3, n))
    zipf_data = zipf_data % np.iinfo(np.uint32).max
    zipf_data.astype("uint32").tofile(f"data/zipf{a}_data_{data_num}_3.dat")
