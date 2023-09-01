import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

def get_data(filename):
    with open(filename,'r') as f:
        data = f.read().split('Test\n')[1]
    lines = data.split('\n')
    X = []
    Y = []
    for line in lines:
        if not line:
            break
        if line == "move finished!":
            continue
        x,y = line.split(',')

        X.append(int(x))
        Y.append(float(y))

    return X,Y


def get_data_from_cs(filename):
    with open(filename,'r') as f:
        data = f.read()
    lines = data.split('\n')
    X = []
    Y = []
    count = 0
    for line in lines:
        if not line:
            continue
        words = line.split(" ")
        if words[0].find("scan_func") >= 0:
            # print(words)
            if len(words) > 2:
                count += 1
                X.append(count)
                Y.append(float(words[1]))
    return X,Y
# fig,ax = plt.subplots()

def draw_append_pic():
    X,Y1 = get_data("/home/antares/nvm/bindextest/bindex1/data.txt")
    # print(X)
    X,Y2 = get_data("/home/antares/nvm/bindextest/bindex2/data.txt")

    plt.plot(X,Y1,label="Bindex1",linestyle='-',marker='',color='black')
    plt.plot(X,Y2,label="Bindex2",linestyle=':',marker='',color='black')

    print(plt.xlim(0,99))
    # print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,99,34)
    y_ticks = np.linspace(0,0.04,10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('data batch')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: append")
    ax.legend()

    plt.show()
    plt.savefig("example.png")

def get_scan_data(filename,local):
    with open(filename,'r') as f:
        datas = f.read().split('\n\n')
    # print(datas[1]) 
    X = []
    CP = []
    RE = []
    LT = []
    count = 0
    for data in islice(datas, 5, None):
        # print(data)
        # print("####################")
        # count += 1
        # if (count == 5):
        #     break
        # continue
        lines = data.split("\n")
        # print(lines)
        if lines[-1] != "CHECK PASSED!":
            print(lines)
            print("Warning: Check not passed!")
            break
        count += 1
        print(lines[local])
        time = lines[local].split(" ")[2]
        X.append(count)
        LT.append(float(time))
        # print(X,LT)
    return X,LT

def get_scan_data_new(filename,local):
    with open(filename,'r') as f:
        lines = f.read().split('\n')
    # print(datas[1]) 
    X = []
    CP = []
    RE = []
    LT = []
    count = 0
    for line in lines:
        words = line.split(" ")
        if len(words) > 3:
            if words[0] == "eq" or words[0] == "lt" or words[0] == "ge":
                time = words[2]
                LT.append(float(time))
                count += 1
                X.append(count)
    return X,LT

def draw_scan_pic(width):
    X,Y1 = get_scan_data("/home/antares/nvm/bindextest/scanTest/bindex1/logs/uniform/UNIFORM_1_selc_100_b_{}.log".format(width),2)
    # print(X)
    X,Y2 = get_scan_data("/home/antares/nvm/bindextest/scanTest/bindex2/logs/uniform/UNIFORM_1_selc_100_b_{}.log".format(width),2)

    plt.plot(X,Y1,label="Bindex1",linestyle='-',marker='',color='black')
    plt.plot(X,Y2,label="Bindex2",linestyle=':',marker='',color='black')

    print(plt.xlim(0,100))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,200,11)
    y_ticks = np.linspace(0,4,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('counts')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()
    # plt.savefig("scan 8.png")

def draw_copy_pic(width):
    X,Y1 = get_scan_data("/home/antares/nvm/bindextest/scanTest/bindex1/logs/uniform/UNIFORM_1_selc_100_b_{}.log".format(width),0)
    # print(X)
    X,Y2 = get_scan_data("/home/antares/nvm/bindextest/scanTest/bindex2/logs/uniform/UNIFORM_1_selc_100_b_{}.log".format(width),0)

    plt.plot(X,Y1,label="Bindex1")
    plt.plot(X,Y2,label="Bindex2")

    print(plt.xlim(0,100))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,200,11)
    y_ticks = np.linspace(0,4,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('counts')
    ax.set_ylabel('use time')
    ax.set_title("Bindex1 & Bindex2: copy width {}".format(width))
    ax.legend()

    plt.show()

def draw_refine_pic(width):
    X,Y1 = get_scan_data("/home/antares/nvm/bindextest/scanTest/bindex1/logs/uniform/UNIFORM_1_selc_100_b_{}.log".format(width),1)
    # print(X)
    X,Y2 = get_scan_data("/home/antares/nvm/bindextest/scanTest/bindex2/logs/uniform/UNIFORM_1_selc_100_b_{}.log".format(width),1)

    plt.plot(X,Y1,label="Bindex1")
    plt.plot(X,Y2,label="Bindex2")

    print(plt.xlim(0,100))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,200,11)
    y_ticks = np.linspace(0,4,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('counts')
    ax.set_ylabel('use time')
    ax.set_title("Bindex1 & Bindex2: refine width {}".format(width))
    ax.legend()

    plt.show()

def draw_diff_pic():
    X,Y1 = get_data("/home/antares/VirtualShareStation/bindex/bindex/str8.txt")
    # print(X)
    X,Y2 = get_data("/home/antares/VirtualShareStation/bindex2/bindex/str8.txt")

    Y3 = []
    for i in range(0,len(Y1)):
        re = 1.0 - Y1[i] / Y2[i]
        Y3.append(re)
    
    plt.plot(X,Y3,label="Diff")
    print(plt.xlim(0,100))
    print(plt.ylim(0,0.04))
    x_ticks = np.linspace(0,100,21)
    y_ticks = np.linspace(0,0.8,10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('data batch')
    ax.set_ylabel('efficiency %')
    # ax.set_title("Bindex1 vs Bindex2")
    ax.legend()

    plt.show()

def draw_test_pic():
    X,Y1 = get_data("/home/antares/VirtualShareStation/bindex/bindex/str8.txt")
    # print(X)
    X,Y2 = get_data("/home/antares/VirtualShareStation/bindex2/bindex/str8.txt")

    plt.plot(X,Y1,label="Bindex1",linestyle='-',marker='',color='black')
    plt.plot(X,Y2,label="Bindex2",linestyle=':',marker='',color='black')


    Y3 = []
    for i in range(0,len(Y1)):
        re = 1.0 - Y1[i] / Y2[i]
        Y3.append(re)

    print(plt.xlim(0,100))
    # print(plt.ylim(0,0.04))
    x_ticks = np.linspace(0,100,21)
    y_ticks = np.linspace(0,0.026,10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('data batch')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 vs Bindex2")
    ax.legend()

    plt.show()
    plt.savefig("example.png")

def draw_block_pic():
    X = (0.5,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192)
    Xz = []
    for i in X:
        Xz.append(str(i))
    Xz = tuple(Xz)
    Y1 = [
        0.0001011,
        0.0001024,
        0.0001009,
        0.0001114,
        0.0001519,
        0.0002041,
        0.0003833,
        0.0008959,
        0.0017831,
        0.0028091,
        0.0046259,
        0.0079463,
        0.0150325,
        0.0285344,
        0.0567348
    ]

    plt.plot(np.arange(15),Y1,label="Bindex1",linestyle='-',marker='^',color='black')

    # print(plt.xlim(0,100))
    print(plt.ylim(0,0.04))
    x_ticks = np.linspace(0,100,21)
    y_ticks = np.linspace(0,0.07,20)
    plt.xticks(np.arange(15),Xz)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('block size (KB)')
    ax.set_ylabel('write time (ms)')
    ax.set_title("Block size test")

    plt.show()
    plt.savefig("example.png")

def get_ldata(filename):
    with open(filename,'r') as f:
        data = f.read()
    lines = data.split('\n')
    X = []
    Y = []
    for line in lines:
        if not line:
            break
        if line[0:7] == "datalen":
            x = line.split(" ")[1]
            X.append(int(x))
        if line[0:6] == "APPEND":
            y = line.split(" ")[3]
            Y.append(float(y))
        
    return X,Y

def draw_ldata_pic():
    X,Y1 = get_ldata("/home/antares/nvm/bindextest/bindex1/ldata.txt")
    X,Y2 = get_ldata("/home/antares/nvm/bindextest/bindex2/ldata.txt")

    Xz = []
    for i in X:
        Xz.append(str(i))
    Xz = tuple(Xz)

    plt.plot(np.arange(12),Y1,label="Bindex1",linestyle='-',marker='',color='black')
    plt.plot(np.arange(12),Y2,label="Bindex2",linestyle=':',marker='',color='black')


    y_ticks = np.linspace(0,3100,20)
    plt.xticks(np.arange(12),Xz)

    ax = plt.gca()


    ax.set_xlabel('data batch')
    ax.set_ylabel('use time (ms)')
    
    ax.legend()

    plt.show()


def draw_unsorted_scan(width):
    X,Y1 = get_scan_data("/home/antares/Code/bindex-unsorted/search_{}.log".format(width),2)
    # print(X)
    X,Y2 = get_scan_data("/home/antares/Code/bindex-baseline-pmdk/search_{}.log".format(width),2)

    plt.plot(X,Y1,label="Bindex-unsorted",linestyle='-',marker='',color='cyan')
    plt.plot(X,Y2,label="Bindex-baseline-pmdk",linestyle='-',marker='',color='red')

    print(plt.xlim(0,100))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,200,11)
    y_ticks = np.linspace(0,6,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('counts')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()

def cal_avg_and_up(X,Y1,Y2,y1_name='',y2_name=''):
    y1_sum = 0
    y2_sum = 0
    for i in X:
        print
        y1_sum += Y1[i - 1]
        y2_sum += Y2[i - 1]
    
    y1_avg = y1_sum / len(Y1)
    y2_avg = y2_sum / len(Y2)

    print("{} AVG: {:.3f}".format(y1_name, y1_avg))
    print("{} AVG: {:.3f}".format(y2_name, y2_avg))

    if y1_avg > y2_avg:
        print("{:.2f}%".format((y1_avg - y2_avg) / y1_avg * 100))
    else:
        print("{:.2f}%".format((y2_avg - y1_avg) / y2_avg * 100))

def draw_scan_3(width):
    X,Y1 = get_scan_data("/home/antares/Code/bindex-unsorted/search_{}.log".format(width),2)
    X,Y2 = get_scan_data("/home/antares/Code/bindex-unsorted-noslot/search_{}.log".format(width),2)
    X,Y3 = get_scan_data("/home/antares/Code/bindex-baseline-pmdk/search_{}.log".format(width),2)
    X,Y4 = get_scan_data("/home/antares/Code/bindex-baseline/search_{}.log".format(width),2)

    # plt.plot(X,Y1,label="Bindex-unsorted-slotarray",linestyle='-',marker='',color='cyan')
    plt.plot(X,Y2,label="Bindex-unsorted-bitmap",linestyle='-',marker='',color='red')
    plt.plot(X,Y3,label="Bindex-baseline-pmdk",linestyle='-',marker='',color='yellow')
    plt.plot(X,Y4,label="Bindex-baseline",linestyle='-',marker='',color='green')

    print("unsorted-bitmapVSbaseline-pmdk")
    cal_avg_and_up(X, Y2, Y3, "unsorted-bitmap", "baseline-pmdk")

    print("unsorted-bitmapVSbaseline")
    cal_avg_and_up(X, Y2, Y4, "unsorted-bitmap", "baseline")

    print(plt.xlim(0,100))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,200,11)
    y_ticks = np.linspace(0,6,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('counts')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()

def mergeData(pointNum,Y):
    result = []
    blockSize = int(len(Y) / pointNum)
    print(blockSize)
    for i in range(0,pointNum):
        blockSum = 0
        for j in range(0,blockSize):
            blockSum += Y[i * blockSize + j]
        result.append(blockSum / blockSize)
    return result


def draw_scan_bitmapVSnotbitmap(width):
    X,Y1 = get_scan_data("/home/antares/Code/bindex-unsorted-noslot/bitmap_search_{}.log".format(width),2)
    X,Y2 = get_scan_data("/home/antares/Code/bindex-unsorted-noslot/search_{}.log".format(width),2)
    X,Y3 = get_scan_data("/home/antares/Code/bindex-baseline-pmdk/search_{}.log".format(width),2)

    Y1 = mergeData(10,Y1)
    Y2 = mergeData(10,Y2)
    Y3 = mergeData(10,Y3)
    X = [1,2,3,4,5,6,7,8,9,10]

    # plt.plot(X,Y1,label="Bindex-unsorted-slotarray",linestyle='-',marker='',color='cyan')
    plt.plot(X,Y1,label="Bindex-unsorted-bitmap",linestyle='-',marker='o',color='red')
    plt.plot(X,Y2,label="Bindex-unsorted-nobitmap",linestyle='-',marker='v',color='blue')
    plt.plot(X,Y3,label="Bindex-baseline-pmdk",linestyle=':',marker='',color='purple')

    print("unsorted-bitmapVSnobitmap")
    cal_avg_and_up(X, Y1, Y2, "unsorted-bitmap", "unsorted-nobitmap")


    print(plt.xlim(0,10))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,10,11)
    y_ticks = np.linspace(0,6,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()


    ax.set_xlabel('counts')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()

def save_scan_data(width,merge=False):
    X,Y1 = get_scan_data("/home/antares/Code/bindex-unsorted-noslot/bitmap_search_{}.log".format(width),2)
    X,Y2 = get_scan_data("/home/antares/Code/bindex-unsorted-noslot/search_{}.log".format(width),2)
    X,Y3 = get_scan_data("/home/antares/Code/bindex-baseline-pmdk/search_{}.log".format(width),2)
    X,Y4 = get_scan_data("/home/antares/Code/bindex/search_{}.log".format(width),2)

    if merge == True:
        Y1 = mergeData(10,Y1)
        Y2 = mergeData(10,Y2)
        Y3 = mergeData(10,Y3)
        Y4 = mergeData(10,Y4)
        X = [1,2,3,4,5,6,7,8,9,10]


    Y_name = [
        "bitmap",
        "nobitmap",
        "baseline-NVM",
        "baseline-DRAM"
    ]
    Y = [Y1, Y2, Y3, Y4]

    with open("scan_data.txt",'w') as f:
        line = "version "
        for i in X:
            line += str(i)
            line += " "
        line += "\n"
        f.write(line)


        for i in range(0,4):
            line = Y_name[i] + " "
            for j in range(0,len(X)):
                line += str(Y[i][j])
                line += " "
            line += "\n"
            f.write(line)
    
    print("[+]Save to 'scan_data.txt' finished")


def draw_tpc_scan():
    X, Y1 = get_scan_data_new("/home/lym/Code/bindex/search_32.log",2)
    X, Y2 = get_scan_data_new("/home/lym/Code/bindex-plus/search_32.log",2)
    # X, Y2 = get_data_from_cs("/home/lym/Code/bindex/tpc_search/tpc_search_batch_cs_2.log")

    Y1 = mergeData(10,Y1)
    Y2 = mergeData(10,Y2)
    X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    # plt.plot(X,Y1,label="Bindex-tpc-date-scan",linestyle='-',marker='o',color='red')
    # plt.plot(X,Y2,label="Cs-tpc-date-scan",linestyle='-',marker='v',color='blue')
    
    # cal_avg_and_up(X,Y1,Y2,"Bindex","Cs")

    X = np.arange(10) + 1
    total_width, n = 0.8 , 2
    width = total_width / n
    X = X - (total_width - width) / 2
    
    plt.bar(X,Y1,width=width,label="Bindex-DRAM-scan",color='orange')
    plt.bar(X+width,Y2,width=width,label="Bindex-NVM-scan",color='cornflowerblue')
    # print(Y1)
    print(plt.xlim(0,10))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,11,12)
    x_name = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    y_ticks = np.linspace(0,5,20)
    plt.xticks(x_ticks,x_name)
    plt.yticks(y_ticks)

    ax = plt.gca()

    ax.set_xlabel('selectivity ')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()
    plt.savefig("./search.png")
# draw_scan_bitmapVSnotbitmap(32)
# save_scan_data(32,True)

draw_tpc_scan()

# draw_scan_3(32)

# draw_unsorted_scan(32)

# draw_diff_pic()
# draw_block_pic()
# draw_copy_pic(32)
# draw_refine_pic(32)
# draw_test_pic()
# draw_append_pic()
# draw_scan_pic(32)
# draw_ldata_pic()
