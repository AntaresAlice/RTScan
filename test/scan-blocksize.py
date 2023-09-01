import time
import timeit
import os
import sys
import re

file1 = ""
def sorted_scan_test(blocksize):
    # time_start = time.time()
    time1 = timeit.timeit()

    logtime = time.strftime("%H-%M-%S-%m-%d-%y")
    # TODO: timeit
    width = 32
    code_width = "width_{}".format(width)

    # compile file

    max_val = 1 << width

    targets = ""

    for selectivity in range(0,110,10):
        target = int((max_val - 1) * selectivity / 100)
        targets += str(target)
        targets += ","
    
    operator = "lt"

    args = "-l {} -o {} ".format(targets,operator)

    output_file = "log/{}-search-sorted-{}-{}.log".format(logtime,width,blocksize)
    file1 = "{}-search-sorted-{}.log".format(logtime,width)
    print(args)

    cmd = "sudo ./bindex -u " + args + " > " + output_file

    print(cmd)

    # os.system("ls")
    os.system(cmd)


    time2 = timeit.timeit()
    # time_end = time.time()

    print("time uesd: ",time2 - time1)

file2 = ""
def unsorted_scan_test(blocksize):
    # time_start = time.time()
    time1 = timeit.timeit()

    logtime = time.strftime("%H-%M-%S-%m-%d-%y")
    # TODO: timeit
    width = 32
    code_width = "width_{}".format(width)

    # compile file

    max_val = 1 << width

    targets = ""

    for selectivity in range(0,110,10):
        target = int((max_val - 1) * selectivity / 100)
        targets += str(target)
        targets += ","
    
    operator = "lt"

    args = "-l {} -o {} ".format(targets,operator)

    output_file = "log/{}-search-unsorted-{}-{}.log".format(logtime,width,blocksize)
    file2 = "{}-search-sorted-{}.log".format(logtime,width)
    print(args)

    cmd = "sudo ./bindex " + args + " > " + output_file

    print(cmd)

    # os.system("ls")
    os.system(cmd)


    time2 = timeit.timeit()
    # time_end = time.time()

    print("time uesd: ",time2 - time1)

note = ""
global output_file_for_analyze
output_file_for_analyze= ""
def pure_scan_test(column_num):
    # time_start = time.time()
    time1 = timeit.timeit()

    logtime = time.strftime("%H-%M-%S-%m-%d-%y")
    # TODO: timeit
    width = 32
    code_width = "width_{}".format(width)
    
    args = "-b {}".format(column_num)
    
    global note
    if note != "":
        note = "-" + note

    output_file = "log/{}-search-pure-{}-{}{}.log".format(logtime,width,column_num,note)
    global output_file_for_analyze
    output_file_for_analyze = output_file
    print(args)

    cmd = "sudo ./bindex " + args + " -f ./uniform_data_1e8_3.dat > " + output_file

    print(cmd)

    # os.system("ls")
    os.system(cmd)


    time2 = timeit.timeit()
    # time_end = time.time()

    print("time uesd: ",time2 - time1)

cmd = sys.argv[1]

os.system("make clean")
os.system("make")
# os.system("sudo ./bindex -c -f ./uniform_data_1e8_3.dat")

if len(sys.argv) < 3:
    print("Usage: scan-blocksize.py <mode> <blockSize/columnNum> <note>")
    exit(0)
    
if len(sys.argv) >= 3:
    note = sys.argv[3]

if cmd == "sorted":
    blocksize = sys.argv[2]
    sorted_scan_test(blocksize)
if cmd == "unsorted":
    blocksize = sys.argv[2]
    unsorted_scan_test(blocksize)
if cmd == "pure":
    column_num = sys.argv[2]
    pure_scan_test(column_num)


def getRefineTimeCount(logfile, countStep = 0.1):
    X = []
    Y = []
    count = 0
    with open(logfile,"r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines:
            words = re.split(r"[ ]+", line)
            if words[0] == "[Time]":
                if words[1] == "refine:":
                    used_time = float(words[2])
                    Y.append(used_time)
                    X.append(count)
                    count += countStep
    return X,Y

def getRaynumCount(logfile, countStep = 0.1):
    X = []
    Y = []
    count = 0
    with open(logfile,"r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines:
            words = re.split(r"[ ]+", line)
            if words[0] == "[OptiX]":
                if words[1] == "launch_width":
                    used_time = float(words[-1])
                    Y.append(used_time)
                    X.append(count)
                    count += countStep
    return X,Y

if output_file_for_analyze != "":
    print(getRefineTimeCount(output_file_for_analyze))
    print(getRaynumCount(output_file_for_analyze))

exit(0)

# sorted_scan_test(blocksize)
# unsorted_scan_test(blocksize)
""" drawcmd = "python3 draw.py 32 {} {}".format(file1,file2)
print(drawcmd)
os.system(drawcmd)
 """