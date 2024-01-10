#include "bindex.h"

map<CODE, CODE> encodeMap[MAX_BINDEX_NUM];

void showEncodeMap()
{
    for (int column_id = 0; column_id < 3; column_id++) {
        for (auto it = encodeMap[column_id].begin(); it != encodeMap[column_id].end(); it++) {
            printf("%u:%u\n", (*it).first, (*it).second);
        }
    }
}

void showReportMap(map<CODE, CODE> &reportMap)
{
    for (auto it = reportMap.begin(); it != reportMap.end(); it++) {
        printf("%u:%u\n", (*it).first, (*it).second);
    }
}

void setNewData(CODE *new_data, CODE *old_data, double intervalPerObject, map<CODE, CODE> &reportMap, 
                int column_id, int thread_id, int data_num) {
    int unit_length = (data_num + THREAD_NUM - 1) / THREAD_NUM;
    int l = thread_id * unit_length;
    int r = l + unit_length;
    if (r > data_num) {
        r = data_num;
    }
    for (int i = l; i < r; i++) {
        int leftBound = encodeMap[column_id][old_data[i]];
        int step = int(intervalPerObject * reportMap[old_data[i]]);
        if (step == 1) {
            new_data[i] = leftBound;
        } else {
            new_data[i] = rand() % (step - 1) + leftBound; // Random dispersion
        }
    }
}

void generateReportMapPerThread(CODE *data, int start_pos, int task_size, map<CODE, CODE> &reportMap, int data_num)
{
    int upbound = start_pos + task_size;
    if (upbound > data_num) {
        upbound = data_num;
    }
    for (int i = start_pos; i < upbound; i++) {
        reportMap[data[i]]++;
    }
}

void setDataPerThread(CODE *new_data, int start_pos, int task_size, int start_val, int intervalPerObject, int data_num)
{
    // Warning: this is a simple mapping function and it ignore the original order in the old dataset
    // It won't lead to any obvious error in the current implementation but should be reconsidered in the future
    // the safe implementation is use encodeMap to map each record
    int upbound = start_pos + task_size;
    if (upbound > data_num) {
        upbound = data_num;
    }

    int val = start_val;
    for (int i = start_pos; i < upbound; i++) {
        new_data[i] = start_val;
        start_val += intervalPerObject;
    }
}

// void encodeDataPerThread(CODE *data, int start_pos, int task_size, int start_val, int intervalPerObject, int data_num, map<CODE, CODE> &encodeMap)
// {
//     // Warning: this is a simple mapping function and it ignore the original order in the old dataset
//     // It won't lead to any obvious error in the current implementation but should be reconsidered in the future
//     // the safe implementation is use encodeMap to map each record
//     int upbound = start_pos + task_size;
//     if (upbound > data_num) {
//         upbound = data_num;
//     }

//     int val = start_val;
//     for (int i = start_pos; i < upbound; i++) {
//         data[i] = start_val;
//         start_val += intervalPerObject;
//     }
// }

void mergeReportMap(map<CODE, CODE> &reportMap, vector<map<CODE,CODE>> &threadReportMap)
{
    int n = threadReportMap.size();
    for (int i = 0; i < n; i++) {
        for(auto p:threadReportMap[i]) {
            reportMap[p.first] += p.second;
        }
    }
}

void moveDataByPosPerThread(CODE *data, CODE *new_data, POSTYPE *pos, int start_pos, int task_size, int data_num)
{
    // reset the data by row_id
    // this may be needed in some order-restricted situation
    int upbound = start_pos + task_size;
    if (upbound > data_num) {
        upbound = data_num;
    }

    for (int i = start_pos; i < upbound; i++) {
        data[pos[i]] = new_data[i];
    }
}


CODE **normalEncode(CODE **initialDataSet, int column_num, CODE encode_min, CODE encode_max, int data_num, 
                    POSTYPE **sorted_pos, CODE **sorted_data) {
    std::thread threads[THREAD_NUM];
    for (int column_id = 0; column_id < column_num; column_id++) {
        std::cerr << "[+] encoding column " << column_id << std::endl;
        printf("[+] encoding column %d\n", column_id);

        // split tasks for each thread
        int threadNum = 20; // Warning: threadNum should be less than THREAD_NUM
        int blockSize = (data_num + threadNum - 1) / threadNum;
        vector<map<CODE,CODE>> threadReportMap(threadNum);
        CODE *data = initialDataSet[column_id];
        POSTYPE *pos = argsort(data, data_num);
        sorted_pos[column_id] = pos;

        for (int threadID = 0; threadID < threadNum; threadID++) {
            threads[threadID] = std::thread(
                generateReportMapPerThread, 
                data, 
                threadID * blockSize, 
                blockSize,
                std::ref(threadReportMap[threadID]),
                data_num
            );
            // generateReportMapPerThread(data, threadID * blockSize, blockSize, threadReportMap[threadID], data_num);
        }

        for (int threadID = 0; threadID < threadNum; threadID++) {
            threads[threadID].join();
        }

        map<CODE, CODE> reportMap;
        mergeReportMap(reportMap, threadReportMap);

        std::cerr << "[+] report Map: " << reportMap.size() << std::endl;
        printf("[+] report Map: %ld\n",reportMap.size());
        
        int intervalPerObject = (double(encode_max) - double(encode_min)) / double(data_num);
        printf("[+] intervalPerObject: %d\n", intervalPerObject);

        int start = encode_min;
        for (auto it = reportMap.begin(); it != reportMap.end(); it++) {
            encodeMap[column_id][it->first] = start;
            start += CODE(intervalPerObject * it->second);
        }
        std::cerr << "[+] set encodeMap" << std::endl;
        printf("[+] set encodeMap\n");

        CODE *newdata = (CODE *)malloc(N * sizeof(CODE));
        sorted_data[column_id] = newdata;

        for (int threadID = 0; threadID < threadNum; threadID++) {
            threads[threadID] = std::thread(
                setDataPerThread, 
                newdata, 
                threadID * blockSize, 
                blockSize,
                encode_min + threadID * blockSize * intervalPerObject,
                intervalPerObject,
                data_num
            );
            // generateReportMapPerThread(data, threadID * blockSize, blockSize, threadReportMap[threadID], data_num);
        }

        for (int threadID = 0; threadID < threadNum; threadID++) {
            threads[threadID].join();
        }

        std::cerr << "[+] reset dataset" << std::endl;
        printf("[+] reset dataset\n\n");

        for (int threadID = 0; threadID < threadNum; threadID++) {
            threads[threadID] = std::thread(
                moveDataByPosPerThread,
                data, 
                newdata, 
                pos,
                threadID * blockSize, 
                blockSize,
                data_num
            );
            // generateReportMapPerThread(data, threadID * blockSize, blockSize, threadReportMap[threadID], data_num);
        }

        for (int threadID = 0; threadID < threadNum; threadID++) {
            threads[threadID].join();
        }

        // free(initialDataSet[column_id]);
        // initialDataSet[column_id] = data;
    }

    return initialDataSet;
}

CODE **normalEncode(CODE **initialDataSet, int column_num, CODE encode_min, CODE encode_max, int data_num) {
    for (int column_id = 0; column_id < column_num; column_id++) {
        printf("[+] encoding column %d\n", column_id);
        CODE *data = initialDataSet[column_id];
        map<CODE, int> reportMap;
        for (int i = 0; i < data_num; i++) {
            if (reportMap.find(data[i]) == reportMap.end()) {
                reportMap[data[i]] = 1;
            }
            else {
                reportMap[data[i]] += 1;
            }
        }
        printf("\n[+] report Map: %ld\n ",reportMap.size());
        
        double intervalPerObject = (static_cast<double>(encode_max) - double(encode_min)) / double(data_num);

        printf("[+] intervalPerObject: %f\n", intervalPerObject);

        int start = encode_min;
        for (map<CODE, int>::iterator it = reportMap.begin(); it != reportMap.end(); it++) {
            encodeMap[column_id][(*it).first] = start;
            start += int(intervalPerObject * (*it).second);
        }

        CODE *newdata = (CODE *)malloc(N * sizeof(CODE));
        for (int i = 0; i < data_num; i++) {
            int leftBound = encodeMap[column_id][data[i]];
            int step = int(intervalPerObject * reportMap[data[i]]);
            newdata[i] = rand() % (step - 1) + leftBound;
        }

        free(initialDataSet[column_id]);
        initialDataSet[column_id] = newdata;
    }
    return initialDataSet;
}

CODE encodeQuery(int column_id, CODE old_query) {
    auto it = encodeMap[column_id].lower_bound(old_query); // 可能这个值不存在(最大时)，那么 second 就是 0，对应值映射不到
    if (it->second == 0 && encodeMap[column_id].lower_bound(old_query >> 1)->second != 0) {
        return (--encodeMap[column_id].end())->second + 1; // 99999999 -> 1e8，使得最后一项性能增加
    }
    return it->second;
}

CODE findKeyByValue(const CODE Val, std::map<CODE, CODE>& map_)
{
    CODE last = 0;
    for (auto p : map_) {
        if (p.second > Val) 
            return last;
        last = p.first;
    }
    return 0;
}

bool ifEncodeEqual(const CODE val1, const CODE val2, int bindex_id)
{
    int key1 = findKeyByValue(val1, encodeMap[bindex_id]);
    int key2 = findKeyByValue(val2, encodeMap[bindex_id]);
    if (key1 == key2) return true;
    return false;
}