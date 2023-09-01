import time
import timeit
import os



targets1 = [19980309, 19980804, 19980408, 19980804, 20010112, 20010112, 20010602, 19990501, 19980318, 19980804, 20010112, 19990201, 19990501, 19980318, 20020530]
targets2 = [19980508, 19980903, 19980601, 19980903, 20010211, 20010212, 20010801, 19990701, 19980616, 19980818, 20010211, 19990403, 19990701, 19980616, 20020729]


eq = [20010713,20010910,20011116,20010616]
lt = [19980408,19980408,19980408,19980408]

def singleTest(target1, target2, data_file, idx):
    time1 = timeit.timeit()

    for width in (32,):
        code_width = "width_{}".format(width)

        # compile file

        
        operator = "bt"

        # args = "-l {} -o {} -s".format(targets,operator)
        args = "-l {} -r {} -o {} -f {}".format(target1,target2,operator,data_file)

        output_file = "./tpc_search/tpc_search_{}.log".format(idx)

        print(args)

        cmd = "sudo ./bindex  " + args + " > " + output_file

        # os.system("ls")
        os.system(cmd)

    time2 = timeit.timeit()

    print("time uesd: ",time2 - time1)



def batchTest(target1, target2, data_file, area_num):
    time1 = timeit.timeit()

    for width in (32,):
        code_width = "width_{}".format(width)

        # compile file

        
        operator = "bt"

        atarget1 = ""
        for t1 in target1:
            atarget1 += str(t1)
            atarget1 += ","
        
        atarget2 = ""
        for t2 in target2:
            atarget2 += str(t2)
            atarget2 += ","

        # args = "-l {} -o {} -s".format(targets,operator)
        args = "-l {} -r {} -o {} -f {}".format(atarget1,atarget2,operator,data_file)

        output_file = "./tpc_search/tpc_search_batch_{}.log".format(area_num)

        print(args)

        cmd = "sudo ./bindex  " + args + " > " + output_file

        # os.system("ls")
        os.system(cmd)

    time2 = timeit.timeit()

    print("time uesd: ",time2 - time1)


def batchEqualTest(target1, data_file, area_num, operator):
    time1 = timeit.timeit()

    for width in (32,):
        code_width = "width_{}".format(width)

        # compile file

        atarget1 = ""
        for t1 in target1:
            atarget1 += str(t1)
            atarget1 += ","
        

        # args = "-l {} -o {} -s".format(targets,operator)
        args = "-l {} -o {} -f {}".format(atarget1,operator,data_file)

        output_file = "./tpc_search/tpc_search_batch_{}.log".format(area_num)

        print(args)

        cmd = "sudo ./bindex  " + args + " >> " + output_file

        # os.system("ls")
        os.system(cmd)

    time2 = timeit.timeit()

    print("time uesd: ",time2 - time1)

""" for i in range(0,len(targets1)):
    t1 = targets1[i]
    t2 = targets2[i]
    data_file = "tpc-ds-dateData"
    singleTest(t1,t2,data_file, i) """

batchTest(targets1,targets2,"tpc-ds-dateData","4_128_512")
batchEqualTest(lt,"tpc-ds-dateData","4_128_512","lt")
batchEqualTest(eq,"tpc-ds-dateData","4_128_512","eq")