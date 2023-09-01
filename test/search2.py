import time
import timeit
import os

# time_start = time.time()
time1 = timeit.timeit()


# TODO: timeit
for width in (8,16,32):
    code_width = "width_{}".format(width)

    # compile file

    max_val = 1 << width

    targets = ""

    for selectivity in range(0,100,10):
        target = int((max_val - 1) * selectivity / 100)
        targets += str(target)
        targets += ","
    
    operator = "lt"

    args = "-l {} -o {} -s".format(targets,operator)

    output_file = "search_{}.log".format(width)

    print(args)

    cmd = "sudo ./bindex " + args + " > " + output_file

    # os.system("ls")
    os.system(cmd)


time2 = timeit.timeit()
# time_end = time.time()

print("time uesd: ",time2 - time1)
