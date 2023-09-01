import os
import re
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

def singleTest(logfile):
    

    # clean output file
    cmd = "echo > {}".format(logfile)

    os.system(cmd)

    for insert_num in range(10,100,10):
        print("insert num: ",insert_num)
        cmd = "sudo ./bindex -i -n {} >> {}".format(insert_num,logfile)
        os.system(cmd)

def getInsertData(logfile):
    X = []
    Y = []
    with open(logfile,"r") as f:
        data = f.read()
        lines = data.split("\n")
        for line in lines:
            words = re.split(r"[ ]+", line)
            if words[0] == "APPEND":
                if not words[1].isnumeric():
                    print("[+]ERROR: get insert num error")
                    print(line)
                    continue
                insert_num = int(words[1])
                used_time = float(words[3])
                Y.append(used_time)
                X.append(insert_num)
    return X,Y

def saveToCSV(csvfile, X, Y):
    with open(csvfile,"w") as f:
        for i in range(0,len(X)):
            x = X[i]
            y = Y[i]
            line = "{} {}\n".format(x,y)
            f.write(line)

def draw(X,Y1):
    # plt.plot(X,Y1,label="Bindex-tpc-date-scan",linestyle='-',marker='o',color='red')
    # plt.plot(X,Y2,label="Cs-tpc-date-scan",linestyle='-',marker='v',color='blue')
    rawX = X
    X = np.arange(len(X)) + 1
    total_width, n = 0.8 , 2
    width = total_width / n
    X = X - (total_width - width) / 2
    
    plt.bar(X,Y1,width=width,label="Bindex-tpc-date-scan",color='orange')
    # plt.bar(X+width,Y2,width=width,label="Cs-tpc-date-scan",color='cornflowerblue')
    # print(Y1)
    print(plt.xlim(0,10))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,24,25)
    y_ticks = np.linspace(0,0.5,20)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    ax = plt.gca()

    ax.set_xlabel('query idx')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()


""" logfile = "log/bindex-DRAM-insert-10-100-1.log"
csvfile = "log/bindex-DRAM-insert-10-100-1.csv"
# singleTest(logfile)
X,Y = getInsertData(logfile)
saveToCSV(csvfile,X,Y) """

X,Y = getInsertData("/home/lym/Code/bindex-baseline-pmdk/log/1-26-bindex-baseline-append-1.log")
saveToCSV("log/1-26-bindex-baseline-append-1.csv",X,Y)

X,Y = getInsertData("/home/lym/Code/bindex-plus/log/1-26-bindex-plus-append-nobatch-1.log")
saveToCSV("log/1-26-bindex-plus-append-nobatch-1.csv",X,Y)

# draw()
# print(X,Y)

