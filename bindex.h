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

#include "timer.h"

#define THREAD_NUM 20
#define MAX_BINDEX_NUM 256

#if !defined(WIDTH_4) && !defined(WIDTH_8) && !defined(WIDTH_12) && !defined(WIDTH_16) && !defined(WIDTH_20) && \
    !defined(WIDTH_24) && !defined(WIDTH_28) && !defined(WIDTH_32)
#define WIDTH_32
#endif

#ifndef VAREA_N
#define VAREA_N 128   // 128
#endif

#ifndef DATA_N
#define DATA_N  1e8   //  1e9
#endif


#define ROUNDUP_DIVIDE(x, n) ((x + n - 1) / n)
#define ROUNDUP(x, n) (ROUNDUP_DIVIDE(x, n) * n)
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

#define BATCH_BLOCK_INSERT 0

/*
  optional code width
*/
#ifdef WIDTH_4
typedef uint8_t CODE;
#define MAXCODE ((1 << 4) - 1)
#define MINCODE 0
#define CODEWIDTH 4
#define LOADSHIFT 4
#endif

#ifdef WIDTH_8
typedef uint8_t CODE;
#define MAXCODE ((1 << 8) - 1)
#define MINCODE 0
#define CODEWIDTH 8
#define LOADSHIFT 0
#endif

#ifdef WIDTH_12
typedef uint16_t CODE;
#define MAXCODE ((1 << 12) - 1)
#define MINCODE 0
#define CODEWIDTH 12
#define LOADSHIFT 4
#endif

#ifdef WIDTH_16
typedef uint16_t CODE;
#define MAXCODE ((1 << 16) - 1)
#define MINCODE 0
#define CODEWIDTH 16
#define LOADSHIFT 0
#endif

#ifdef WIDTH_20
typedef uint32_t CODE;
#define MAXCODE ((1 << 20) - 1)
#define MINCODE 0
#define CODEWIDTH 20
#define LOADSHIFT 12
#endif

#ifdef WIDTH_24
typedef uint32_t CODE;
#define MAXCODE ((1 << 24) - 1)
#define MINCODE 0
#define CODEWIDTH 24
#define LOADSHIFT 8
#endif

#ifdef WIDTH_28
typedef uint32_t CODE;
#define MAXCODE ((1 << 28) - 1)
#define MINCODE 0
#define CODEWIDTH 28
#define LOADSHIFT 4
#endif

#ifdef WIDTH_32
typedef uint32_t CODE;
#define MAXCODE ((1L << 32) - 1)
#define MINCODE 0
#define CODEWIDTH 32
#define LOADSHIFT 0
#endif


#define DEBUG_TIME_COUNT 1
#define DEBUG_INFO 1

// VALUES: 0 (uniform, normal), 1 (zipf)
#ifndef DISTRIBUTION
#define DISTRIBUTION 0
#endif

#ifndef ENCODE
#define ENCODE 0
#endif

#define RANGE_SELECTIVITY_11

typedef int POSTYPE;  // Data type for positions
// typedef int CODE;           // Codes are stored as int
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

// const int CODEWIDTH = sizeof(CODE) * 8;
const int BITSWIDTH = sizeof(BITS) * 8;
const int BITSSHIFT = 5;  // x / BITSWIDTH == x >> BITSSHIFT
const int SIMD_ALIGEN = 32;
const int SIMD_JOB_UNIT = 8;  // 8 * BITSWIDTH == __m256i

const int blockInitSize = 3276;  // 2048  // TODO: tunable parameter for optimization
const int blockMaxSize = 4096; // blockInitSize * 2;
const int K = VAREA_N;  // Number of virtual areas
const int N = (int)DATA_N;
// const int MAXCODE = INT_MAX;
// const int MINCODE = INT_MIN;
const int blockNumMax = (N / (K * blockInitSize)) * 4;
// const int blockNumMax = 2;

extern Timer timer;

extern int prefetch_stride;

extern std::vector<CODE> target_numbers_l;  // left target numbers
extern std::vector<CODE> target_numbers_r;  // right target numbers

extern BITS *result1;
extern BITS *result2;

extern CODE *current_raw_data;
extern std::mutex bitmapMutex;

extern std::mutex scan_refine_mutex;
extern int scan_refine_in_position;
extern CODE scan_selected_compares[MAX_BINDEX_NUM][2];
extern bool scan_skip_refine;
extern bool scan_skip_other_face[MAX_BINDEX_NUM];
extern bool scan_skip_this_face[MAX_BINDEX_NUM];
extern CODE scan_max_compares[MAX_BINDEX_NUM][2];  // this special should be renamed to 'common/normal' lol
extern bool scan_inverse_this_face[MAX_BINDEX_NUM];

extern int density_width;
extern int density_height;

extern int default_ray_segment_num;

extern map<CODE, CODE> encodeMap[MAX_BINDEX_NUM];

typedef struct {
  // Struct for a position block
  POSTYPE *pos;  // Position array in a block
  CODE *val;     // Sorted codes in a block TODO: needed to be
  // removed in future implementation for saving
  // space (emmmm... actually we don't need
  // to remove it for experimental evaluations).
  int length;
} pos_block;

typedef struct {
  pos_block *blocks[blockNumMax];
  int blockNum;
  int length;
} Area;

typedef struct {
  Area *areas[K];
  CODE areaStartValues[K];
  BITS *filterVectors[K - 1];
  BITS *filterVectorsInGPU[K - 1];
  POSTYPE area_counts[K];  // Counts of values contained in the first i areas
  POSTYPE length;
  CODE data_min;
  CODE data_max;
} BinDex;

inline int bits_num_needed(int n) {
  // calculate the number of bits for storing n 0-1 bit results
  return ((n - 1) / BITSWIDTH) + 1;
}

void copy_filter_vector_in_GPU(BinDex *bindex, BITS *dev_bitmap, int k, bool negation = false);
void raw_scan(BinDex *bindex, BITS *bitmap, CODE target1, CODE target2, OPERATOR OP, CODE *raw_data, BITS* compare_bitmap = NULL);

CODE **normalEncode(CODE **initialDataSet, int column_num, CODE encode_min, CODE encode_max, int data_num);
CODE encodeQuery(int column_id, CODE old_query);
bool ifEncodeEqual(const CODE val1, const CODE val2, int bindex_id);
CODE findKeyByValue(const CODE Val, std::map<CODE, int>& map_);

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void showGPUInfo();
cudaError_t refineWithCuda(CODE* bitmap, unsigned int size);
cudaError_t GPUbitAndWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitOrWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitCopyWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitCopyNegationWithCuda(BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int n);
cudaError_t GPUbitCopySIMDWithCuda(BITS* result, BITS* dev_bitmap_a, BITS* dev_bitmap_b, unsigned int bitnum);

void initializeOptix(CODE **raw_data, int length, int density_width, int density_height, int column_num, CODE* range, 
                     uint32_t cube_width);
// ! ray_length is 'int' type which may be out of bound when it is too large.
void refineWithOptix(BITS *dev_result_bitmap, double *predicate, int column_num,
                     int ray_length = -1, int ray_segment_num = 10, bool inverse = false, int direction = 2, int ray_mode = 0);
#endif