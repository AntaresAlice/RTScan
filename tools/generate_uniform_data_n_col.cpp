#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <random>
#include <string>

#define MAXCODE ((1L << 32) - 1)
#define MINCODE 0

typedef uint32_t CODE;

CODE **generateUniformData(int data_len, int col_num)
{
    CODE **initial_data;
    initial_data = (CODE **)malloc(col_num * sizeof(CODE *));
    for (int col_id = 0; col_id < col_num; col_id++) {
        initial_data[col_id] = (CODE *)malloc(data_len * sizeof(CODE));
        CODE *data = initial_data[col_id];

        printf("[INFO] initing uniform data %d\n",col_id);
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<CODE> dist(MINCODE, MAXCODE);
        CODE mask = ((uint64_t)1 << (sizeof(CODE) * 8)) - 1;
        for (int i = 0; i < data_len; i++) {
            data[i] = dist(mt) & mask;
            assert(data[i] <= mask);
        }
        printf("[INFO] col %d  first num: %u  last num: %u\n",col_id, initial_data[col_id][0], initial_data[col_id][data_len - 1]);
    }
    return initial_data;
}

void saveDataToFile(CODE** data, int data_len, int col_num) 
{
    //save data
    char DATA_PATH2[256] = "./savefile\0";
    FILE *fp2;
    if (!(fp2 = fopen(DATA_PATH2, "wb")))
    {
        printf("[ERROR] save_to_file: fopen(%s) faild\n", DATA_PATH2);
        exit(-1);
    }
    printf("[INFO] saving data to %s\n", DATA_PATH2);

    for (int col_id = 0; col_id < col_num; col_id++) {
        CODE *col_data = data[col_id];
        if (fwrite(col_data, sizeof(CODE), data_len, fp2) == 0)
        {
            printf("[ERROR] save_to_file col %d: fwrite faild.\n",col_id);
            exit(-1);
        }
    }
    
    fclose(fp2);
    printf("[+] save finished!\n");
}

void checkSavedData(int data_len, int col_num)
{
    char DATA_PATH[256] = "./savefile\0";
    FILE *fp;

    if (!(fp = fopen(DATA_PATH, "rb")))
    {
        printf("init_data_from_file: fopen(%s) faild\n", DATA_PATH);
        exit(-1);
    }
    printf("initing data from %s\n", DATA_PATH);

    CODE *file_data = (CODE *)malloc(data_len * sizeof(CODE));
    for (int col_id = 0; col_id < col_num; col_id++) {
        if (fread(file_data, sizeof(CODE), data_len, fp) == 0)
        {
            printf("init_data_from_file: fread faild.\n");
            exit(-1);
        }
        CODE first_num = file_data[0];
        CODE last_num = file_data[data_len-1];
        printf("[CHECK] col %d  first num: %u  last num: %u\n",col_id, first_num, last_num);
    }
    free(file_data);
}

CODE **getSavedData(char *DATA_PATH, int data_len, int col_num)
{
    FILE *fp;
    // char DATA_PATH[256] = "./uniform_data_1e8_3.dat\0";
    if (!(fp = fopen(DATA_PATH, "rb")))
    {
        printf("init_data_from_file: fopen(%s) faild\n", DATA_PATH);
        exit(-1);
    }
    printf("initing data from %s\n", DATA_PATH);

    CODE **initial_data;
    initial_data = (CODE **)malloc(col_num * sizeof(CODE *));

    for (int col_id = 0; col_id < col_num; col_id++) {
        initial_data[col_id] = (CODE *)malloc(data_len * sizeof(CODE));
        CODE *file_data = initial_data[col_id];
        if (fread(file_data, sizeof(CODE), data_len, fp) == 0)
        {
            printf("init_data_from_file: fread faild.\n");
            exit(-1);
        }
        CODE first_num = file_data[0];
        CODE last_num = file_data[data_len-1];
        printf("[CHECK] col %d  first num: %u  last num: %u\n",col_id, first_num, last_num);
    }

    return initial_data;
}

int main(int argc, char *argv[])
{
    char opt;
    int col_num = 0;
    int data_len = 1e8;
    while ((opt = getopt(argc, argv, "hc:n:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            printf(
                "Usage: %s \n"
                "[-c <col num>]"
                "[-n <data len>] \n",
                argv[0]);
            exit(0);
        case 'n':
            // DATA_N
            data_len = atoi(optarg);
            break;
        case 'c':
            col_num = atoi(optarg);
            break;
        default:
            printf("Error: unknown option %c\n", (char)opt);
            exit(-1);
        }
    }

    assert(col_num > 0);
    assert(data_len > 0);

    printf("[INFO] generate uniform  data len: %d col num: %d\n", data_len, col_num);

    char path[] = "./uniform_data_1e8_3.dat";
    // getSavedData(path,data_len,col_num);
    // exit(0);
    CODE **initial_data = generateUniformData(data_len, col_num);
    saveDataToFile(initial_data, data_len, col_num);
    checkSavedData(data_len, col_num);
}