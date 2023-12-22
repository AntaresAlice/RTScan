#ifndef BINDEX_H_
#define BINDEX_H_

#include <assert.h>
#include <immintrin.h>
#include <limits.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <parallel/algorithm>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <fstream>
#include <malloc.h>
#include <unordered_map>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "timer.h"

#define THREAD_NUM 20
#define MAX_BINDEX_NUM 256

#ifndef VAREA_N
#define VAREA_N 128   // 128
#endif

#ifndef DATA_N
#define DATA_N  1e8   //  1e9
#endif

#define DEBUG_TIME_COUNT 1
#define DEBUG_INFO 1

// VALUES: 0 (uniform, normal), 1 (zipf)
#ifndef DISTRIBUTION
#define DISTRIBUTION 0
#endif

#ifndef ENCODE
#define ENCODE 1
#endif

#define RANGE_SELECTIVITY_11

#ifndef ONLY_DATA_SIEVING
#define ONLY_DATA_SIEVING 0
#endif

#ifndef ONLY_REFINE
#define ONLY_REFINE 0
#endif

#define PRINT_EXCECUTION_TIME(msg, code)           \
  do {                                             \
    struct timeval t1, t2;                         \
    double elapsed;                                \
    gettimeofday(&t1, NULL);                       \
    do {                                           \
      code;                                        \
    } while (0);                                   \
    gettimeofday(&t2, NULL);                       \
    elapsed = (t2.tv_sec - t1.tv_sec) * 1000.0;    \
    elapsed += (t2.tv_usec - t1.tv_usec) / 1000.0; \
    printf(msg " time: %f ms\n", elapsed);         \
  } while (0);

#define CODEWIDTH 32

typedef uint32_t CODE;
typedef int POSTYPE;  // Data type for positions
typedef unsigned int BITS;  // 32 0-1 bit results are stored in a BITS
enum OPERATOR {
  EQ = 0,  // x == a
  LT,      // x < a
  GT,      // x > a
  LE,      // x <= a
  GE,      // x >= a
  BT,      // a < x < b
};

const int RUNS = 1;
const int BITSWIDTH = sizeof(BITS) * 8;
const int BITSSHIFT = 5;  // x / BITSWIDTH == x >> BITSSHIFT
const int SIMD_ALIGEN = 32;
const int SIMD_JOB_UNIT = 8;  // 8 * BITSWIDTH == __m256i
const int K = VAREA_N;  // Number of virtual areas
const int N = (int)DATA_N;

extern Timer timer;
extern map<CODE, CODE> encodeMap[MAX_BINDEX_NUM];

typedef struct {
  CODE areaStartValues[K];
  BITS *filterVectors[K - 1];
  BITS *filterVectorsInGPU[K - 1];
  POSTYPE length;
  CODE data_min;
  CODE data_max;
} BinDex;

template <typename T>
POSTYPE *argsort(const T *v, POSTYPE n) {
  POSTYPE *idx = (POSTYPE *)malloc(n * sizeof(POSTYPE));
  for (POSTYPE i = 0; i < n; i++) {
    idx[i] = i;
  }
  __gnu_parallel::sort(idx, idx + n, [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
POSTYPE *argsort(const T *v, POSTYPE n, POSTYPE *idx) {
  for (POSTYPE i = 0; i < n; i++) {
    idx[i] = i;
  }
  __gnu_parallel::sort(idx, idx + n, [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

inline int bits_num_needed(int n) {
  // calculate the number of bits for storing n 0-1 bit results
  return ((n - 1) / BITSWIDTH) + 1;
}

CODE **normalEncode(CODE **initialDataSet, int column_num, CODE encode_min, CODE encode_max, int data_num, POSTYPE **sorted_pos, CODE **sorted_data);
CODE **normalEncode(CODE **initialDataSet, int column_num, CODE encode_min, CODE encode_max, int data_num);
CODE encodeQuery(int column_id, CODE old_query);
bool ifEncodeEqual(const CODE val1, const CODE val2, int bindex_id);
CODE findKeyByValue(const CODE Val, std::map<CODE, int>& map_);

cudaError_t GPUbitAndWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitCopyWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitCopyNegationWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitCopySIMDWithCuda(BITS* result, BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum);

void initializeOptix(CODE **raw_data, int length, int density_width, int density_height, int column_num, CODE* range, 
                     int cube_width);
void refineWithOptix(BITS *dev_result_bitmap, double *predicate, int column_num,
                     int ray_length = -1, int ray_segment_num = 10, bool inverse = false, int direction = 2, int ray_mode = 0);

void initializeOptixRTc3(CODE **raw_data, int length, int density_width, int density_height, int column_num, 
                         CODE *range, uint32_t cube_width, int direction);
void refineWithOptixRTc3(BITS *dev_result_bitmap, double *predicate, int column_num, 
                            int ray_length = -1, int ray_segment_num = 10, bool inverse = false, int direction = 2, int ray_mode = 0);

void initializeOptixRTc1(CODE **raw_data, int length, int density_width, int density_height, int column_num, CODE* range, int cube_width_factor,
                         int ray_interval, int prim_size);
void refineWithOptixRTc1(BITS *dev_result_bitmap, double *predicate, int column_num,
                         int ray_length = -1, int ray_segment_num = 10, bool inverse = false, int direction = 2, int ray_mode = 0);

void initializeOptixRTScan_2c(CODE **raw_data, int length, int density_width, int density_height, int column_num = 3);
void refineWithOptixRTScan_2c(BITS *dev_result_bitmap, double *predicate, unsigned *range, 
                              int column_num, int ray_segment_num = 1, bool inverse = false);

void initializeOptixRTScan_interval_spacing(CODE **raw_data, int length, int density_width, int density_height, int column_num, CODE* range, double ray_interval_ratio);
void refineWithOptixRTScan_interval_spacing(BITS *dev_result_bitmap, double *predicate, int column_num,
                                            int ray_length, int ray_segment_num, bool inverse, int direction, int ray_mode,
                                            double ray_distance_ratio);
#endif