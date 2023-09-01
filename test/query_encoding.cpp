#include <cstdio>
#include <iostream>
#include <string>

#include "../bindex.h"
using namespace std;

// extern map<CODE, int> encodeMap[MAX_BINDEX_NUM];

void getDataFromFile(char* DATA_PATH, CODE** initial_data, int bindex_num) {
    FILE* fp;

    if (!(fp = fopen(DATA_PATH, "rb"))) {
        printf("init_data_from_file: fopen(%s) faild\n", DATA_PATH);
        exit(-1);
    }
    printf("initing data from %s\n", DATA_PATH);

    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
        initial_data[bindex_id] = (CODE*)malloc(N * sizeof(CODE));
        CODE* data = initial_data[bindex_id];
        // 8/16/32 only
        if (CODEWIDTH == 8) {
            uint8_t* file_data = (uint8_t*)malloc(N * sizeof(uint8_t));
            if (fread(file_data, sizeof(uint8_t), N, fp) == 0) {
                printf("init_data_from_file: fread faild.\n");
                exit(-1);
            }
            for (int i = 0; i < N; i++)
                data[i] = file_data[i];
            free(file_data);
        } else if (CODEWIDTH == 16) {
            uint16_t* file_data = (uint16_t*)malloc(N * sizeof(uint16_t));
            if (fread(file_data, sizeof(uint16_t), N, fp) == 0) {
                printf("init_data_from_file: fread faild.\n");
                exit(-1);
            }
            for (int i = 0; i < N; i++)
                data[i] = file_data[i];
            free(file_data);
        } else if (CODEWIDTH == 32) {
            uint32_t* file_data = (uint32_t*)malloc(N * sizeof(uint32_t));
            if (fread(file_data, sizeof(uint32_t), N, fp) == 0) {
                printf("init_data_from_file: fread faild.\n");
                exit(-1);
            }
            for (int i = 0; i < N; i++)
                data[i] = file_data[i];
            free(file_data);
        } else {
            printf("init_data_from_file: CODE_WIDTH != 8/16/32.\n");
            exit(-1);
        }
        printf("[CHECK] col %d  first num: %u  last num: %u\n", bindex_id, initial_data[bindex_id][0], initial_data[bindex_id][N - 1]);
    }
}

void generate_range_queries(vector<CODE>& target_lower,
                            vector<CODE>& target_upper,
                            vector<string>& search_cmd,
                            int column_num) {
    int NUM_QUERIES = 11;
    CODE span = UINT32_MAX / (NUM_QUERIES - 1);
    target_lower.resize(NUM_QUERIES * column_num);
    search_cmd.resize(NUM_QUERIES * column_num);
    for (int i = 0; i < NUM_QUERIES; i++) {
        for (int column_id = 0; column_id < column_num; column_id++) {
            target_lower[i * column_num + column_id] = i * span;
            search_cmd[i * column_num + column_id] = "lt";
        }
    }
}

int main() {
    Timer timer;
    char data_path[] = "/home/wzm/bindex-raytracing/data/uniform_data_1e8_3.dat";
    CODE* initial_data[MAX_BINDEX_NUM];
    int bindex_num = 3;
    getDataFromFile(data_path, initial_data, bindex_num);

    timer.commonGetStartTime(1);
    printf("[+] remapping initial data...\n");
    normalEncode(initial_data, bindex_num, 0, N, N);
    printf("[+] remap initial data done.\n");
    timer.commonGetEndTime(1);

    vector<CODE> target_lower;
    vector<CODE> target_upper;
    vector<string> search_cmd;
    int NUM_QUERIES = 11;
    target_lower.resize(NUM_QUERIES * bindex_num);
    target_upper.resize(NUM_QUERIES * bindex_num);
    search_cmd.resize(NUM_QUERIES * bindex_num);
    generate_range_queries(target_lower, target_upper, search_cmd, bindex_num);
    for (int i = 0; i < NUM_QUERIES; i++) {
        for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
            timer.commonGetStartTime(16);
            target_upper[i * bindex_num + bindex_id] = encodeQuery(bindex_id, target_upper[i * bindex_num + bindex_id]);
            timer.commonGetEndTime(16);
        }
        printf("[Time] encode query [%d]: %lf ms\n", i, timer.time[16]);
        timer.time[16] = 0.0;
    }

    timer.showTime();
    timer.clear();
    return 0;
}