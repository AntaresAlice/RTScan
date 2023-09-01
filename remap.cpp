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

CODE **normalEncode(CODE **initialDataSet, int column_num, CODE encode_min, CODE encode_max, int data_num)
{
    for (int column_id = 0; column_id < column_num; column_id++) {
        printf("[+] encoding column %d\n", column_id);
        CODE *data = initialDataSet[column_id];
        map<CODE, CODE> reportMap;
        for (int i = 0; i < data_num; i++) {
            if (reportMap.find(data[i]) == reportMap.end()) {
                reportMap[data[i]] = 1;
            }
            else {
                reportMap[data[i]] += 1;
            }
        }
        printf("[+] report Map: %ld\n",reportMap.size());
        
        double intervalPerObject = (static_cast<double>(encode_max) - double(encode_min)) / double(data_num);
        printf("[+] intervalPerObject: %lf\n", intervalPerObject);

        int start = encode_min;
        for (auto it = reportMap.begin(); it != reportMap.end(); it++) {
            encodeMap[column_id][it->first] = start;
            start += CODE(intervalPerObject * it->second);
        }
        printf("[+] set encodeMap\n");

        CODE *newdata = (CODE *)malloc(N * sizeof(CODE));
        unordered_map<CODE,int> hash;
        for (int i = 0; i < data_num; i++) {
            int leftBound = encodeMap[column_id][data[i]];
            int step = int(intervalPerObject * reportMap[data[i]]);
            if (hash.count(leftBound) > 0) 
                newdata[i] = rand() % (step - 1) + leftBound;
            else {
                newdata[i] = leftBound;
                hash[leftBound] = 1;
            }
        }
        // std::thread threads[THREAD_NUM];
        // for (int i = 0; i < THREAD_NUM; i++) {
        //     threads[i] = std::thread(setNewData, newdata, data, intervalPerObject, ref(reportMap), 
        //                              column_id, i, data_num);
        // }
        // for (int i = 0; i < THREAD_NUM; i++) {
        //     threads[i].join();
        // }
        printf("[+] reset dataset\n\n");

        free(initialDataSet[column_id]);
        initialDataSet[column_id] = newdata;
    }

    return initialDataSet;
}


CODE encodeQuery(int column_id, CODE old_query) {
    auto it = encodeMap[column_id].lower_bound(old_query);
    // printf("(encodeQuery: {%u,%u})", it->first, it->second);
    return (*it).second;
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