import time
import timeit
import os


def codewidth_scan():
    # time_start = time.time()
    time1 = timeit.timeit()

    logtime = time.strftime("%H-%M-%S-%m-%d-%y")
    # TODO: timeit
    for width in (8,16,32):
        code_width = "width_{}".format(width)

        # compile file

        max_val = 1 << width

        targets = ""

        for selectivity in range(0,110,10):
            target = int((max_val - 1) * selectivity / 100)
            targets += str(target)
            targets += ","
        
        operator = "lt"

        args = "-l {} -o {} -s".format(targets,operator)

        output_file = "log/{}-scan-1e9-{}.log".format(logtime,width)

        print(args)

        cmd = "sudo ./bindex " + args + " > " + output_file

        # os.system("ls")
        os.system(cmd)


    time2 = timeit.timeit()
    # time_end = time.time()

    print("time uesd: ",time2 - time1)


def normal_scan():
    # time_start = time.time()
    time1 = timeit.timeit()

    logtime = time.strftime("%H-%M-%S-%m-%d-%y")
    # TODO: timeit
    max_val = 1 << 32

    targets = ""

    for selectivity in range(0,110,10):
        target = int((max_val - 1) * selectivity / 100)
        targets += str(target)
        targets += ","
    
    operator = "lt"

    args = "-l {} -o {} -s ".format(targets,operator)

    output_file = "log/{}-scan-1e9.log".format(logtime)

    print(args)

    cmd = "sudo ./bindex -f ../bindex-plus/data_1e9.dat " + args + " > " + output_file

    print(cmd)

    # os.system("ls")
    os.system(cmd)


    time2 = timeit.timeit()
    # time_end = time.time()

    print("time uesd: ",time2 - time1)

normal_scan()