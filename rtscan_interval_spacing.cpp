#include "rt.h"

std::mutex scan_refine_mutex;
int scan_refine_in_position;
CODE scan_selected_compares[MAX_BINDEX_NUM][2];
bool scan_skip_refine;
bool scan_skip_other_face[MAX_BINDEX_NUM];
bool scan_skip_this_face[MAX_BINDEX_NUM];
CODE scan_max_compares[MAX_BINDEX_NUM][2];
bool scan_inverse_this_face[MAX_BINDEX_NUM];

int density_width;
int density_height;

int default_ray_segment_num = 64;
int ray_length = -1;
int ray_mode   = 0; // (0, cont), (1, space)

double ray_interval_ratio;
double ray_distance_ratio;

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
        if (fread(data, sizeof(uint32_t), N, fp) == 0) {
            printf("init_data_from_file: fread faild.\n");
            exit(-1);
        }
        printf("[CHECK] col %d  first num: %u  last num: %u\n", bindex_id, initial_data[bindex_id][0], initial_data[bindex_id][N - 1]);
    }
}

int main(int argc, char* argv[]) {
    printf("N = %d\n", N);
    char opt;
    int selectivity;
    char DATA_PATH[256] = "\0";
    char SCAN_FILE[256] = "\0";
    int bindex_num = 3;

    density_width = 1200;
    density_height = 1200;  // maximum ray-width and maximum ray-height
    default_ray_segment_num = 64;

    // get command line options
    bool TEST_INSERTING = false;

    while ((opt = getopt(argc, argv, "hf:a:b:c:d:e:w:m:s:p:t:")) != -1) {
        switch (opt) {
            case 'h':
                printf(
                    "Usage: %s \n"
                    "[-f <input-file>]\n"
                    "[-w <ray-range-width>] [-m <ray-range-height>]\n"
                    "[-s <ray-segment-num>]\n"
                    "[-p <scan-predicate-file>]\n"
                    "[-a <ray-length>]\n"
                    "[-t <primitive-type> - 0: triangle, 1: sphere, 2: aabb]\n"
                    "[-c <ray-interval> - ratio]\n"
                    "[-d <ray-distance> - ratio]\n",
                    argv[0]);
                exit(0);
            case 'f':
                strcpy(DATA_PATH, optarg);
                break;
            case 'b':
                bindex_num = atoi(optarg);
                break;
            case 's':
                default_ray_segment_num = atoi(optarg);
                break;
            case 'w':
                density_width = atoi(optarg);
                break;
            case 'm':
                density_height = atoi(optarg);
                break;
            case 'p':
                strcpy(SCAN_FILE, optarg);
                break;
            case 'a':
                ray_length = atoi(optarg);
                break;
            case 'c':
                ray_interval_ratio = atof(optarg);
                break;
            case 'd':
                ray_distance_ratio = atof(optarg);
                break;
            case 'e':
                ray_mode = atoi(optarg);
                break;
            default:
                printf("Error: unknown option %c\n", (char)opt);
                exit(-1);
        }
    }
    assert(bindex_num >= 1);

    CODE* initial_data[MAX_BINDEX_NUM];
    getDataFromFile(DATA_PATH, initial_data, bindex_num);

    CODE* range = (CODE*)malloc(sizeof(CODE) * 6);
    for (int i = 0; i < bindex_num; i++) {
        range[i * 2] = 14;
        range[i * 2 + 1] = 4294967282;
    }
    Timer timer;
    initializeOptixRTScan_interval_spacing(initial_data, N, density_width, density_height, 3, range, ray_interval_ratio);

    double** compares = (double**)malloc(bindex_num * sizeof(double*));
    double* dev_predicate = (double*)malloc(bindex_num * 2 * sizeof(double));
    for (int i = 0; i < bindex_num; i++) {
        compares[i] = &(dev_predicate[i * 2]);
    }

    // set compare here

    // 0.2 858993458.000000 872396308.000000 69.000000 858993459.000000 42.000000 858993459.000000
    // compares[0][0] = 858993458;
    // compares[0][1] = 872396308;
    // compares[1][0] = 69;
    // compares[1][1] = 858993459;
    // compares[2][0] = 42;
    // compares[2][1] = 858993459;

    // 0.5 2147475444.000000 2147483647.000000 69.000000 2147483647.000000 42.000000 2147483647.000000 
    // compares[0][0] = 2147475444;
    // compares[0][1] = 2147483647;
    // compares[1][0] = 69;
    // compares[1][1] = 2147483647;
    // compares[2][0] = 42;
    // compares[2][1] = 2147483647;
    
    // 0.7 14.000000 3019827290.000000 69.000000 3019741350.000000 3006477105.000000 3019731858.000000
    // compares[0][0] = 14;
    // compares[0][1] = 3019827290;
    // compares[1][0] = 69;
    // compares[1][1] = 3019741350;
    // compares[2][0] = 3006477105;
    // compares[2][1] = 3019731858;

    // 0.8
    compares[0][0] = 14;
    compares[0][1] = 3422511049;
    compares[1][0] = 69;
    compares[1][1] = 3422436046;
    compares[2][0] = 3422596912;
    compares[2][1] = 3435973836;

    // 0.9
    // compares[0][0] = 14;
    // compares[0][1] = 3858639816;
    // compares[1][0] = 69;
    // compares[1][1] = 3858820032;
    // compares[2][0] = 3858882078;
    // compares[2][1] = 3865470565;

    // malloc dev bitmap for result after refine
    BITS* dev_bitmap_for_refined_result;
    int bitmap_len = bits_num_needed(N);
    cudaMalloc(&dev_bitmap_for_refined_result, bitmap_len * sizeof(BITS));
    cudaMemset(dev_bitmap_for_refined_result, 0x0, bitmap_len * sizeof(BITS));

    int direction = 1;

    // warmup
    refineWithOptixRTScan_interval_spacing(dev_bitmap_for_refined_result, dev_predicate, bindex_num, ray_length, default_ray_segment_num, false, direction, ray_mode, ray_distance_ratio);

    timer.commonGetStartTime(13);
    int runs = 20;
    for (int i = 0; i < runs; i++) {
        refineWithOptixRTScan_interval_spacing(dev_bitmap_for_refined_result, dev_predicate, bindex_num, ray_length, default_ray_segment_num, false, direction, ray_mode, ray_distance_ratio);
    }
    timer.commonGetEndTime(13);
    timer.time[13] /= runs;
    timer.showTime();
}