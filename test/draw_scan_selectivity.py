import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

def draw_tpc_scan():
    # Y1 = [13.14, 17.03, 18.02, 17.79, 17.16, 16.47, 16.82, 17.46, 17.96, 16.97, 13.88]  # 12-42-14-10-28-22-search-pure-32-3.log
    Y1 = [4.80, 8.08, 8.66, 8.77, 8.19, 7.52, 8.05, 8.66, 8.64, 8.07, 5.46] # 16-01-30-10-31-22-search-pure-32-3.log
    # Y2 = [3.73, 12.13, 20.52, 29.96, 39.41, 49.92, 60.38, 73, 87.70, 101.32, 119.26]
    # Y2 = [3.73, 4.78, 6.86, 8.98, 12.12, 22.60, 32.05, 44.65, 66.70, 95.05, 121.29]
    # Y2 = [8.34204,10.4429,11.488,13.5889,16.74,27.281,36.678,49.2729,70.2959,99.655,126.947]   # old  10-28 with all
    # Y2 = [0.046875,2.050049,4.318848,6.423096,8.860107,19.425781,29.243164,40.585938,62.717041,92.24707,118.955811] # 10-31 pure without trans
    # trans = [0.956055,0.957031,0.957031,0.957031,0.956787,0.957031,0.955811,0.957031,0.958008,0.958008,0.958008]
    Y2 = [1.014160,  1.319092,  1.831055,  2.976075,  5.440185, 11.142090, 17.403808, 27.658936, 41.182129, 60.125000, 80.418945]  # 11-08 RT
    Y3 = [0.115, 0.127, 0.125, 0.126, 0.126, 0.126, 0.133, 0.126, 0.126, 0.126, 0.125] # 16-31-43-12-02-22-search-pure-32-3.log  bindex-scan cuda without refine
    # for i in range(0, len(Y2)):
    #     Y2[i] = Y2[i] + trans[i]
    # plt.plot(X,Y1,label="Bindex-tpc-date-scan",linestyle='-',marker='o',color='red')
    # plt.plot(X,Y2,label="Cs-tpc-date-scan",linestyle='-',marker='v',color='blue')
    
    # cal_avg_and_up(X,Y1,Y2,"Bindex","Cs")

    X = np.arange(11)
    total_width, n = 0.8 , 2
    width = total_width / n
    X = X - (total_width - width) / n
    
    plt.bar(X,Y1,width=width,label="Bindex-DRAM-scan",color='cornflowerblue')
    plt.bar(X+width,Y2,width=width,label="RT-scan",color='orange')
    # print(Y1)
    print(plt.xlim(-0.5,10.5))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,10,11)
    x_name = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    y_ticks = np.linspace(0,100,21)
    plt.xticks(x_ticks,x_name)
    plt.yticks(y_ticks)

    ax = plt.gca()

    ax.set_xlabel('selectivity ')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()
    plt.savefig("./pic/Bindex-RT-11-08-with-trans.png")

def draw_tpc_scan_line():
    # Y1 = [13.14, 17.03, 18.02, 17.79, 17.16, 16.47, 16.82, 17.46, 17.96, 16.97, 13.88]  # 12-42-14-10-28-22-search-pure-32-3.log
    Y1 = [4.80, 8.08, 8.66, 8.77, 8.19, 7.52, 8.05, 8.66, 8.64, 8.07, 5.46] # 16-01-30-10-31-22-search-pure-32-3.log
    # Y2 = [3.73, 12.13, 20.52, 29.96, 39.41, 49.92, 60.38, 73, 87.70, 101.32, 119.26]
    # Y2 = [3.73, 4.78, 6.86, 8.98, 12.12, 22.60, 32.05, 44.65, 66.70, 95.05, 121.29]
    # Y2 = [8.34204,10.4429,11.488,13.5889,16.74,27.281,36.678,49.2729,70.2959,99.655,126.947]   # old  10-28 with all
    # Y2 = [0.046875,2.050049,4.318848,6.423096,8.860107,19.425781,29.243164,40.585938,62.717041,92.24707,118.955811] # 10-31 pure without trans
    # trans = [0.956055,0.957031,0.957031,0.957031,0.956787,0.957031,0.955811,0.957031,0.958008,0.958008,0.958008]
    # Y2 = [1.014160,  1.319092,  1.831055,  2.976075,  5.440185, 11.142090, 17.403808, 27.658936, 41.182129, 60.125000, 80.418945]  # 11-08 RT
    # Y3 = [0.115, 0.127, 0.125, 0.126, 0.126, 0.126, 0.133, 0.126, 0.126, 0.126, 0.125] # 16-31-43-12-02-22-search-pure-32-3.log  bindex-scan cuda without refine
    Y3 = [0.059, 0.855, 0.602051, 1.192871, 1.213135, 1.281982, 1.396973, 1.651123, 1.994873, 2.258057, 0.276855] # bindex/log/12-58-26-12-12-22-search-pure-32-3.log
    Y4 = [1.422, 1.552, 1.410156, 1.460938, 1.427979, 1.386963, 1.666016, 1.402832, 1.391113, 1.384033, 1.384033]

    for i in range(0, len(Y3)):
        Y3[i] = Y3[i] + Y4[i]

    X = np.arange(11)
    plt.plot(X,Y1,label="Bindex-DRAM-scan",linestyle='-',marker=None,color='cornflowerblue')
    # plt.plot(X,Y2,label="RT-scan",linestyle='-',marker=None,color='orange')
    plt.plot(X,Y3,label="Bindex-CUDA-scan",linestyle='-',marker=None,color='purple')

    # total_width, n = 0.6 , 2
    # width = total_width / n
    # X = X - (total_width - width) / n
    
    # plt.bar(X,Y3,width=width,label="Bindex-CUDA-scan",color='purple')
    # plt.bar(X+width,Y1,width=width,label="Bindex-DRAM-scan",color='cornflowerblue')
    # plt.bar(X+width*2,Y2,width=width,label="RT-scan",color='orange')
    # print(Y1)
    print(plt.xlim(-0.5,10.5))
    print(plt.ylim(0,0.1))
    x_ticks = np.linspace(0,10,11)
    x_name = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    y_ticks = np.linspace(0,10, 11)
    plt.xticks(x_ticks,x_name)
    plt.yticks(y_ticks)

    ax = plt.gca()

    ax.set_xlabel('selectivity ')
    ax.set_ylabel('use time (ms)')
    # ax.set_title("Bindex1 & Bindex2: Scan width {}".format(width))
    ax.legend()

    plt.show()
    plt.savefig("./pic/Bindex-CUDA-12-12-refine.png")

draw_tpc_scan_line()