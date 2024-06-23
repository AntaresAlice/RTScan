import time
import os
import sys
import numpy as np

def RTScan_3c_unique_1e8():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/column3/unique1e8/{logtime}-RTScan.log"
    os.system('make clean')
    os.system(
        f'make rtscan DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE=0 BUILD_TYPE=Release')
    # ray_length = 1e8 / 90 = 1111111
    args = f'-b 3 -w 1200 -m 1200 -a 1111111 -z 1 -q 11 -p test/scan_cmd_1e8-3c.txt'
    cmd = f"./bin/rtscan {args} >> {output_file}"
    print(cmd)
    os.system(cmd)

def RTScan_1c_unique_1e8():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/column1/unique1e8/{logtime}-RTScan-1c.log"
    os.system('make clean')
    os.system(
        f'make rtc1 DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 PRIMITIVE_TYPE=1 BUILD_TYPE=Release')
    density_list = [5e6] # approximate optimal configuration obtained from the experiment
    for density in density_list:
        args = f'-b 3 -w {density} -m {density} -s 1 -a -1 -u 1 -p test/scan_cmd_1e8-1c-rtscan.txt'
        cmd = f"./bin/rtc1 {args} >> {output_file}"
        print(cmd)
        os.system(cmd)
        
def RTScan_2c_unique_1e8():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/column2/unique1e8/{logtime}-RTScan-2c.log"
    os.system('make clean')
    os.system(f'make rtscan_2c DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    density_list = [10000] # approximate optimal configuration obtained from the experiment
    for density in density_list:
        args = f'-b 2 -w {density} -m {density} -s 1 -p test/scan_cmd_1e8-2c.txt'
        cmd = f"./bin/rtscan_2c {args} >> {output_file}"
        print(cmd)
        os.system(cmd)

def RTScan_3c_2p32_encode_1e8():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/memory/{logtime}-RTScan-2p32-encoding.log"
    os.system('make clean')
    os.system(
        f'make rtscan DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE=1 BUILD_TYPE=Release')
    # ray_length = 1e8 / 90 = 1111111
    args = f'-b 3 -w 1200 -m 1200 -a 1111111 -z 1 -q 11 -p test/scan_cmd_32_3c.txt -f data/uniform_data_1e8_3.dat'
    cmd = f"./bin/rtscan {args} >> {output_file}"
    print(cmd)
    os.system(cmd)

def RTScan_3c_zipf_encode_1e8():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/column3/memory/{logtime}-RTScan-zipf1.5-encoding.log"
    os.system('make clean')
    os.system(
        f'make rtscan DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE=1 BUILD_TYPE=Release')
    # ray_length = 1e8 / 90 = 1111111
    args = f'-b 3 -w 1200 -m 1200 -a 1111111 -z 1 -q 1 -p test/zipf1.5.txt -f data/zipf1.5_data_1e8_3.dat'
    cmd = f"./bin/rtscan {args} >> {output_file}"
    print(cmd)
    os.system(cmd)
    
def RTScan_3c_skewed(encode=1):
    logtime = time.strftime("%y%m%d-%H%M%S")
    input_file_list = [
        'data/zipf1.1_data_1e8_3.dat',
        'data/zipf1.3_data_1e8_3.dat',
        'data/zipf1.5_data_1e8_3.dat',
        'data/normal_data_1e8_3.dat',
        ]
    scan_file_list = [
        'test/zipf1.1.txt',
        'test/zipf1.3.txt',
        'test/zipf1.5.txt',
        'test/normal.txt',
        ]
    os.system('make clean')
    os.system(f'make rtscan DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE={encode} BUILD_TYPE=Release')
    output_file = f"log/column3/skew/{logtime}-1e8-RTScan-skew_selec0.9-encode{encode}.log"
    for i in range(len(input_file_list)):
        # args = f'-b 3 -w 1200 -m 1200 -a 1120000 -y 1 -q 1 -p {scan_file_list[i]}'
        args = f'-b 3 -w 1200 -m 1200 -a 1111111 -y 1 -q 1 -p {scan_file_list[i]}'
        cmd = f"./bin/rtscan {args} -f {input_file_list[i]} >> {output_file}"
        print(cmd)
        os.system(cmd)

def RTScan_3c_2p6():
    os.system('make clean')
    os.system(f'make rtscan DATA_N=1e8 VAREA_N=16 SMALL_DATA_RANGE=1 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    data_range_list = [63]
    direction_list = [1] # wide(0)/narrow(1) face
    density_list = [ # approximate optimal configuration obtained from the experiment
        [150],
    ]
    segment_num_list = [ # approximate optimal configuration obtained from the experiment
        [80],
    ]
    logtime = time.strftime("%y%m%d-%H%M%S")
    for k in range(len(data_range_list)):
        output_file = f"log/column3/small_range/{logtime}-2p6-narrow-sieve_16-adjust_interval_spacing-i150_s80.log"
        for direction in direction_list:
            for density in density_list[k]:
                for segment in segment_num_list[k]:
                    args = f'-b 3 -a -2 -s {segment} -m {density} -w {density} -d {data_range_list[k]},{data_range_list[k]},{data_range_list[k]} -z {direction} -v -1 -y 1 -q 11 -g 1 -p test/scan_cmd_2p6.txt'
                    cmd = "./bin/rtscan " + args + " >> " + output_file
                    print(cmd)
                    os.system(cmd)

def RTScan_3c_2p6_encoding_old():
    vector_num = 16
    os.system('make clean')
    os.system(f'make rtscan DATA_N=1e8 VAREA_N={vector_num} ENCODE=1 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    data_range_list = [63]
    direction_list = [1] # wide(0)/narrow(1) face
    density_list = [ # approximate optimal configuration obtained from the experiment
        [600],
    ]
    segment_num_list = [ # approximate optimal configuration obtained from the experiment
        [100],
    ]
    logtime = time.strftime("%y%m%d-%H%M%S")
    for k in range(len(data_range_list)):
        output_file = f"log/column3/small_range/{logtime}-2p6-encode-narrow-sieve_{vector_num}.log"
        for direction in direction_list:
            for density in density_list[k]:
                for segment in segment_num_list[k]:
                    args = f'-b 3 -a -2 -s {segment} -m {density} -w {density} -d {data_range_list[k]},{data_range_list[k]},{data_range_list[k]} -z {direction} -y 1 -q 11 -g 1 -p test/scan_cmd_2p6.txt'
                    cmd = "./bin/rtscan " + args + " >> " + output_file
                    print(cmd)
                    os.system(cmd)
                    
def RTScan_3c_2p6_encoding():
    vector_num = 16
    os.system('make clean')
    os.system(f'make rtscan DATA_N=1e8 VAREA_N={vector_num} ENCODE=1 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    data_range_list = [63]
    direction_list = [1] # wide(0)/narrow(1) face
    density_list = [ # approximate optimal configuration obtained from the experiment
        [1200],
    ]
    logtime = time.strftime("%y%m%d-%H%M%S")
    for k in range(len(data_range_list)):
        output_file = f"log/column3/small_range/{logtime}-2p6-encode-narrow-sieve_{vector_num}-i1200.log"
        for direction in direction_list:
            for density in density_list[k]:
                args = f'-b 3 -a -2 -m {density} -w {density} -d {data_range_list[k]},{data_range_list[k]},{data_range_list[k]} -z {direction} -y 1 -q 11 -g 1 -p test/scan_cmd_2p6.txt'
                cmd = "./bin/rtscan " + args + " >> " + output_file
                print(cmd)
                os.system(cmd)

def RTScan_3c_sieving_vectors_64():
    sieve_list = [64]
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/column3/sieving_vectors/{logtime}-1e8-3.log"
    for sieve in sieve_list:
        os.system('make clean')
        os.system(f'make rtscan DATA_N=1e8 VAREA_N={sieve} PRIMITIVE_TYPE=2 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 ENCODE=0')
        args = f'-b 3 -w 1200 -m 1200 -a 1111111 -z 1 -q 11 -p test/scan_cmd_1e8-3c.txt'
        cmd = f"./bin/rtscan {args} >> {output_file}"
        print(cmd)
        os.system(cmd)

def RTScan_3c_vary_sieving_vectors():     
    sieve_list = [32, 64, 128]
    logtime = time.strftime("%y%m%d-%H-%M-%S")
    output_file = f"log/column3/sieving_vectors/{logtime}-1e8-3-vary-sieve32to128-info.log"
    for sieve in sieve_list:
        os.system('make clean')
        os.system(f'make rtscan DATA_N=1e8 VAREA_N={sieve} PRIMITIVE_TYPE=2 DEBUG_ISHIT_CMP_RAY=1 DEBUG_INFO=1 DISTRIBUTION=0')
        args = f"-b 3 -w 1200 -m 1200 -a 48000000 -e 1 -c 1 -q 11 -p test/scan_cmd_32_3c.txt"
        cmd = f"./bin/rtscan {args} -f data/uniform_data_1e8_3.dat >> {output_file}"
        print(cmd)
        os.system(cmd)
        
def RTScan_3c_vary_interval_spacing():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/column3/interval_spacing/{logtime}-1e8-3-6M-interval_and_distance.log"
    os.system('make clean')
    os.system(f'make rtscan_interval_spacing DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 PRIMITIVE_TYPE=2 BUILD_TYPE=Release')
    ray_interval_ratio_list = [0.25, 0.5, 0.75, 1]
    ray_spacing_ratio_list = [0, 0.25, 0.5, 0.75, 1]
    for interval_ratio in ray_interval_ratio_list:
        for spacing_ratio in ray_spacing_ratio_list:
            # 48000000 6000000
            args = f'-b 3 -w 1200 -m 1200 -a 6000000 -e 1 -c {interval_ratio} -d {spacing_ratio}'
            cmd = f"./bin/rtscan_interval_spacing {args} -f data/uniform_data_1e8_3.dat >> {output_file}" 
            print(cmd)
            os.system(cmd)

def BinDexCUDA_uniform(column_num=3):
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system(f'make bindex_cuda DATA_N=1e8 ONLY_REFINE=0 ONLY_DATA_SIEVING=0')
    output_file = f"log/cuda/column{column_num}/{logtime}-1e8-{column_num}c-cuda.log"
    args = f"-b {column_num} -p test/scan_cmd_32_{column_num}c.txt"
    data_file = 'data/uniform_data_1e8_3.dat' if column_num <= 3 else 'data/uniform_data_1e8_4.dat'
    cmd = f"./bin/bindex_cuda {args} -f {data_file} >> {output_file}"
    print(cmd)
    os.system(cmd)

def BinDexCUDA_skewed():
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system(f'make bindex_cuda DATA_N=1e8 ONLY_REFINE=0 ONLY_DATA_SIEVING=0')
    input_file_list = ['data/zipf1.1_data_1e8_3.dat',
                       'data/zipf1.3_data_1e8_3.dat',
                       'data/zipf1.5_data_1e8_3.dat',
                       'data/normal_data_1e8_3.dat']
    scan_file_list = ['test/zipf1.1.txt',
                      'test/zipf1.3.txt',
                      'test/zipf1.5.txt',
                      'test/normal.txt']
    output_file = f"log/cuda/skew/{logtime}-1e8-3c-skew_selec0.9.log"
    for j in range(len(input_file_list)):
        cmd = f"./bin/bindex_cuda -b 3 -p {scan_file_list[j]} -f {input_file_list[j]} >> {output_file}"
        print(cmd)
        os.system(cmd)

def BinDexCUDA_index_overhead(): # same as BinDex
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system(f'make bindex_cuda DATA_N=1e8 ONLY_REFINE=0 ONLY_DATA_SIEVING=0')
    output_file = f"log/cuda/index_overhead/{logtime}-cuda-zipf1.5.log"
    args = f"-b 3 -p test/zipf1.5.txt -f data/zipf1.5_data_1e8_3.dat"
    cmd = f"./bin/bindex_cuda {args} >> {output_file}"
    print(cmd)
    os.system(cmd)

def BinDex_uniform_2p32(column_num=3):
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system('make bindex DATA_N=1e8 VAREA_N=128')
    output_file = f"log/bindex/{logtime}-uniform2p32-{column_num}c.log"
    args = f"-b {column_num} -p test/scan_cmd_32_{column_num}c.txt"
    cmd = f"./bin/bindex {args} -f data/uniform_data_1e8_3.dat >> {output_file}" 
    print(cmd)
    os.system(cmd)

def BinDex_skewed():
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system(f'make bindex DATA_N=1e8 VAREA_N=128')
    input_file_list = ['data/zipf1.1_data_1e8_3.dat',
                       'data/zipf1.3_data_1e8_3.dat',
                       'data/zipf1.5_data_1e8_3.dat',
                       'data/normal_data_1e8_3.dat']
    scan_file_list = ['test/zipf1.1.txt',
                      'test/zipf1.3.txt',
                      'test/zipf1.5.txt',
                      'test/normal.txt']
    output_file = f"log/bindex/{logtime}-skew_selec0.9.log"
    for j in range(len(input_file_list)):
        cmd = f"./bin/bindex -b 3 -p {scan_file_list[j]} -f {input_file_list[j]} >> {output_file}"
        print(cmd)
        os.system(cmd)

def RTc3_2p32():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/rtc3/{logtime}-1e8-3-2p32.log"
    ray_density_list = [400] # approximate optimal configuration obtained from the experiment
    os.system('make clean')
    os.system(f'make rtc3 DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    for density in ray_density_list:
        args = f'-b 3 -w {density} -m {density} -s 1 -a -1 -z 1 -p test/scan_cmd_32_3c.txt -q 11'
        cmd = f"./bin/rtc3 {args} -f data/uniform_data_1e8_3.dat >> {output_file}"
        print(cmd)
        os.system(cmd)
        
def RTc3_skewed():
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system(f'make rtc3 DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    input_file_list = ['data/zipf1.1_data_1e8_3.dat',
                       'data/zipf1.3_data_1e8_3.dat',
                       'data/zipf1.5_data_1e8_3.dat',
                       'data/normal_data_1e8_3.dat']
    scan_file_list = ['test/zipf1.1.txt',
                      'test/zipf1.3.txt',
                      'test/zipf1.5.txt',
                      'test/normal.txt']
    density_list = [
        [400],
    ]
    data_range_list = [4294967295]
    for i in range(len(data_range_list)):
        output_file = f"log/rtc3/{logtime}-1e8-3-2p32-400-skew_selec0.9.log"
        for density in density_list[i]:
            for j in range(len(input_file_list)):
                args = f'-b 3 -w {density} -m {density} -s 1 -a -1 -v {data_range_list[i]},{data_range_list[i]},{data_range_list[i]} -z 1 -p {scan_file_list[j]} -q 1'
                cmd = f"./bin/rtc3 {args} -f {input_file_list[j]} >> {output_file}"
                print(cmd)
                os.system(cmd)

def RTc3_2p6():
    logtime = time.strftime("%y%m%d-%H%M%S")
    os.system('make clean')
    os.system(f'make rtc3 DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 BUILD_TYPE=Release')
    density_list = [ # approximate optimal configuration obtained from the experiment
        [400],
    ]
    data_range_list = [63]
    data_range_exp = {63:'2^6', 127:'2^7', 200:'200', 300:'300', 255:'2^8', 511:'2^9', 1023:'2^10', 16383:'2^14', 262143:'2^18', 1048575:'2^20', 4194303:'2^22', 67108863:'2^26', 4294967295: '2^32'}
    for i in range(len(data_range_list)):
        output_file = f"log/small_data_range/RTc3/{data_range_exp[data_range_list[i]]}/{logtime}-1e8-RTc3-data_range_{data_range_exp[data_range_list[i]]}-i_400.log"
        for density in density_list[i]:
            print(density)
            args = f'-b 3 -w {density} -m {density} -s 1 -a -1 -v {data_range_list[i]},{data_range_list[i]},{data_range_list[i]} -z 0 -q 11 -g 1'
            # cmd = "./bindex " + args + " -f data/uniform_data_1e8_3.dat >> " + output_file
            cmd = "./bin/rtc3 " + args + " >> " + output_file
            print(cmd)
            os.system(cmd)

def RTc1_1c_2p32(): # use `eq` to scan one column
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/rtc1/{logtime}-1e8-1-2p32.log"
    ray_density = [18000] # approximate optimal configuration obtained from the experiment
    os.system('make clean')
    os.system('make rtc1 DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 PRIMITIVE_TYPE=0 BUILD_TYPE=Release')
    for i in range(len(ray_density)):
        args = f'-b 3 -w {ray_density[i]} -m {ray_density[i]} -s 1 -a -1 -u 0 -p test/scan_cmd_32_1c-rtc1.txt'
        cmd = "./bin/rtc1 " + args + " -f data/uniform_data_1e8_3.dat >> " + output_file
        print(cmd)
        os.system(cmd)

def RTc1_skewed():
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/rtc1/{logtime}-1e8-1-skew_selec0.9.log"
    input_file_list = ['data/zipf1.1_data_1e8_3.dat',
                       'data/zipf1.3_data_1e8_3.dat',
                       'data/zipf1.5_data_1e8_3.dat',
                       'data/normal_data_1e8_3.dat']
    scan_file_list = ['test/zipf1.1-rtc1.txt',
                      'test/zipf1.3-rtc1.txt',
                      'test/zipf1.5-rtc1.txt',
                      'test/normal-rtc1.txt']
    os.system('make clean')
    os.system('make rtc1 DATA_N=1e8 DEBUG_ISHIT_CMP_RAY=0 DEBUG_INFO=0 DISTRIBUTION=0 PRIMITIVE_TYPE=0 BUILD_TYPE=Release')
    for i in range(len(input_file_list)):
        args = f'-b 3 -w 18000 -m 18000 -s 1 -a -1 -u 0 -q 1 -p {scan_file_list[i]}'
        cmd = f"./bin/rtc1 {args} -f {input_file_list[i]} >> {output_file}"
        print(cmd)
        os.system(cmd)


# RTScan
# RTScan_1c_unique_1e8()
# RTScan_2c_unique_1e8()
# RTScan_3c_unique_1e8()
RTScan_3c_skewed(encode=1) # ENCODE
# RTScan_3c_skewed(encode=0)
# RTScan_3c_2p6()
# RTScan_3c_2p6_encoding() # ENCODE

# RTScan_3c_2p32_encode_1e8() # ENCODE
# RTScan_3c_zipf_encode_1e8() # ENCODE

# RTScan_3c_vary_sieving_vectors()
# RTScan_3c_vary_interval_spacing()


# RTc3
# RTc3_2p32()
# RTc3_skewed()
# RTc3_2p6()

# RTc1
# Only scan one column a time. The time for scanning three columns is the sum of the time for scanning one column three times, 
# plus the time taken to merge result bit vectors for each column.
# RTc1_1c_2p32()
# RTc1_skewed()


# BinDex_CUDA
# BinDexCUDA_uniform(column_num=3)
# BinDexCUDA_uniform(column_num=1)
# BinDexCUDA_uniform(column_num=2)
# BinDexCUDA_uniform(column_num=4)
# BinDexCUDA_skewed()
# BinDexCUDA_index_overhead()

# BinDex
# BinDex_uniform_2p32(column_num=1)
# BinDex_uniform_2p32(column_num=2)
# BinDex_uniform_2p32(column_num=3)
# BinDex_uniform_2p32(column_num=4)
# BinDex_skewed()