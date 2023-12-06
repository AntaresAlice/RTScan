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


#include <iostream>
#include <vector>
#include <regex>
#include <fstream>


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
#include <mutex>

using namespace std;

// time count part
class Timer{
  public:

  double time[30];
  mutex timeMutex[30];
  double timebase;

  Timer() {
    for (int i = 0; i < 30; i++) {
      time[i] = 0.0;
    }
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
  }

  void clear() {
    for (int i = 0; i < 30; i++) {
      time[i] = 0.0;
    }
  }
  
  void showTime() {
    cout << endl;
    cout << "###########   Time  ##########" << endl;
    cout << "[Time] build bindex: ";
    cout << time[0] << " ms" << endl;
    // cout << "[Time] append to bindex: ";
    // cout << time[1] << endl;
    // cout << "[Time] insert to block: ";
    // cout << time[2] << endl;
    // cout << "[Time] split block: ";
    // cout << time[3] << endl;
    // cout << "[Time] insert to area: ";
    // cout << time[4] << endl;
    // cout << "[Time] init new space: ";
    // cout << time[5] << endl;
    // cout << "[Time] append to filter vector: ";
    // cout << time[6] << endl;
    cout << "[Time] total time: ";
    cout << time[11] << " ms" << endl;
    cout << "##############################" << endl;
    cout << endl;
  }


  void commonGetStartTime(int timeId) {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    lock_guard<mutex> lock(timeMutex[timeId]);
    time[timeId] -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }

  void commonGetEndTime(int timeId) {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    lock_guard<mutex> lock(timeMutex[timeId]);
    time[timeId] += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }
};

Timer timer;

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

const int blockInitSize = 3276;  // 2048  
const int blockMaxSize = 4096; // blockInitSize * 2;
const int K = VAREA_N;  // Number of virtual areas
const int N = (int)DATA_N;
// const int MAXCODE = INT_MAX;
// const int MINCODE = INT_MIN;
const int blockNumMax = (N / (K * blockInitSize)) * 4;
int prefetch_stride = 6;

std::vector<CODE> target_numbers_l;  // left target numbers
std::vector<CODE> target_numbers_r;  // right target numbers

BITS *result1;
BITS *result2;

CODE *current_raw_data;
char scan_file[256];


std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::string s;
    s.append(1, delim);
    std::regex reg(s);
    std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   std::sregex_token_iterator());
    return elems;
}

void display_bitmap(BITS *bitmap, int bitmap_len);

template <typename T>
POSTYPE *argsort(const T *v, POSTYPE n) {
  POSTYPE *idx = (POSTYPE *)malloc(n * sizeof(POSTYPE));
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

BITS gen_less_bits(const CODE *val, CODE compare, int n) {
  // n must <= BITSWIDTH (32)
  BITS result = 0;
  for (int i = 0; i < n; i++) {
    if (val[i] < compare) {
      result += (1 << (BITSWIDTH - 1 - i));
    }
  }
  return result;
}

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
  BITS *filterVectors[K - 1];
  POSTYPE area_counts[K];  // Counts of values contained in the first i areas
  POSTYPE length;
} BinDex;

void init_pos_block(pos_block *pb, CODE *val_f, POSTYPE *pos_f, int n) {
  assert(n <= blockInitSize);
  pb->length = n;
  pb->pos = (POSTYPE *)malloc(blockMaxSize * sizeof(POSTYPE));
  pb->val = (CODE *)malloc(blockMaxSize * sizeof(CODE));
  for (int i = 0; i < n; i++) {
    pb->pos[i] = pos_f[i];
    pb->val[i] = val_f[i];
  }
}

CODE block_start_value(pos_block *pb) { return pb->val[0]; }

int insert_to_block(pos_block *pb, CODE *val_f, POSTYPE *pos_f, int n) {
  // Insert max(n, #vacancy) elements to a block, return 0 if the block is still
  // not filled up.
  if (DEBUG_TIME_COUNT) timer.commonGetStartTime(2);
  int flagNum, length_new;
  if ((n + pb->length) >= blockMaxSize) {
    // Block filled up! This block will be splitted in insert_to_area(..)
    flagNum = blockMaxSize - pb->length;  // The number of successfully inserted
    // elements, flagNum will be return
    length_new = blockMaxSize;
  } else {
    flagNum = 0;
    length_new = pb->length + n;
  }

  int k, i, j;
  // Merge two sorted array
  for (k = length_new - 1, i = pb->length - 1, j = length_new - pb->length - 1; i >= 0 && j >= 0;) {
    if (val_f[j] >= pb->val[i]) {
      pb->val[k] = val_f[j];
      pb->pos[k--] = pos_f[j--];
    } else {
      pb->val[k] = pb->val[i];
      pb->pos[k--] = pb->pos[i--];
    }
  }
  while (j >= 0) {
    pb->val[k] = val_f[j];
    pb->pos[k--] = pos_f[j--];
  }
  pb->length = length_new;

  if (DEBUG_TIME_COUNT) timer.commonGetEndTime(2);
  return flagNum;
}

int insert_to_block_without_val(pos_block *pb, CODE *val_f, POSTYPE *pos_f, int n, CODE *raw_data) {
  // Insert max(n, #vacancy) elements to a block, return 0 if the block is still
  // not filled up.
  if (DEBUG_TIME_COUNT) timer.commonGetStartTime(2);
  int flagNum, length_new;
  if ((n + pb->length) >= blockMaxSize) {
    // Block filled up! This block will be splitted in insert_to_area(..)
    flagNum = blockMaxSize - pb->length;  // The number of successfully inserted
    // elements, flagNum will be return
    length_new = blockMaxSize;
  } else {
    flagNum = 0;
    length_new = pb->length + n;
  }

  if (BATCH_BLOCK_INSERT) {
    int k, i, j;
    // Merge two sorted array
    for (k = length_new - 1, i = pb->length - 1, j = length_new - pb->length - 1; i >= 0 && j >= 0;) {
      if (val_f[j] >= raw_data[pb->pos[i]]) {
        pb->pos[k--] = pos_f[j--];
      } else {
        pb->pos[k--] = pb->pos[i--];
      }
    }
    while (j >= 0) {
      pb->pos[k--] = pos_f[j--];
    }
  }
  else {
      // Separate insert to block
      int insert_num = (flagNum > 0)?flagNum:n;
      for (int j = 0; j < insert_num; j++){
        int noflag = 0;
        for (int i = 0; i < pb->length; i++) {
          if (raw_data[pb->pos[i]] > val_f[j]) {
            noflag = 1;
            for (int k = pb->length; k > i; k--) {
              pb->pos[k] = pb->pos[k-1];
            }
            pb->pos[i] = pos_f[j];
            pb->length = pb->length + 1;
            break;
          }
        }
        if (!noflag) {
          pb->pos[pb->length] = pos_f[j];
          pb->length = pb->length + 1;
        }
      }
    }




  pb->length = length_new;

  if (DEBUG_TIME_COUNT) timer.commonGetEndTime(2);
  return flagNum;
}

void init_area(Area *area, CODE *val, POSTYPE *pos, int n) {
  // TODO: An area may explode for extremely skewed data.
  // Area containing only unique code should be considered in
  // future implementation
  int i = 0;
  area->blockNum = 0;
  area->length = n;
  while (i + blockInitSize < n) {
    area->blocks[area->blockNum] = (pos_block *)malloc(sizeof(pos_block));
    init_pos_block(area->blocks[area->blockNum], val + i, pos + i, blockInitSize);
    (area->blockNum)++;
    i += blockInitSize;
  }
  area->blocks[area->blockNum] = (pos_block *)malloc(sizeof(pos_block));
  init_pos_block(area->blocks[area->blockNum], val + i, pos + i, n - i);
  area->blockNum++;
  assert(area->blockNum <= blockNumMax);
}

CODE area_start_value(Area *area) { return area->blocks[0]->val[0]; }

void area_split_block(Area *area, int block_idx) {
  if (DEBUG_TIME_COUNT) timer.commonGetStartTime(3);
  assert(area->blockNum < blockNumMax);
  pos_block *pb_old = area->blocks[block_idx];
  pb_old->length = blockInitSize;  // Split pb_old into two blocks, only keep
  // half of the original values in pb_old

  // Fill values into new block
  pos_block *pb_new = (pos_block *)malloc(sizeof(pos_block));
  init_pos_block(pb_new, pb_old->val + blockInitSize, pb_old->pos + blockInitSize, blockMaxSize - blockInitSize);

  // Update blocks in area
  for (int i = area->blockNum; i > (block_idx + 1); i--) {
    area->blocks[i] = area->blocks[i - 1];
  }
  area->blocks[block_idx + 1] = pb_new;
  area->blockNum++;
  if (DEBUG_TIME_COUNT) timer.commonGetEndTime(3);
}

void insert_to_area(Area *area, CODE *val, POSTYPE *pos, int n, CODE *raw_data) {
  // TODO: Inserting too many elements into an area will explode (the blocks
  // array will be filled up), automatically rebuilding the whole BinDex
  // structure (or enlarging current area) is needed in future implementation

  int i, j;
  for (i = 0, j = 0; i < n && j < area->blockNum - 1;) {
    int num_insert_to_block = 0;
    int start = i;
    while (val[i] < block_start_value(area->blocks[j + 1]) && i < n) {
      i++;
      num_insert_to_block++;
    }
    if (num_insert_to_block) {
      int flagNum = insert_to_block_without_val(area->blocks[j], val + start, pos + start, num_insert_to_block, raw_data);
      while (flagNum) {
        area_split_block(area, j++);
        num_insert_to_block -= flagNum;
        start += flagNum;
        flagNum = insert_to_block_without_val(area->blocks[j], val + start, pos + start, num_insert_to_block, raw_data);
      }
    }
    j++;
  }

  if (i < n) {  // Insert val[i] to the end of last block if val[i] is no less
    // than the maximum value of current area
    int num_insert_to_block = n - i;
    int start = i;
    int flagNum = insert_to_block_without_val(area->blocks[j], val + start, pos + start, num_insert_to_block, raw_data);
    while (flagNum) {
      area_split_block(area, j++);
      num_insert_to_block -= flagNum;
      start += flagNum;
      flagNum = insert_to_block_without_val(area->blocks[j], val + start, pos + start, num_insert_to_block, raw_data);
    }
  }
  area->length += n;

}

void show_volume(Area *area)
{
    cout << "[+] Area  volume:" << endl;
    for (int i = 0; i < area->blockNum; i++)
    {
        cout << "Block " << i << ": " << "start value: " << area->blocks[i]->val[0] << " Size: " << area->blocks[i]->length << endl;
    }
}

void display_block(pos_block *pb) {
  printf("Virtual space values:\t");
  for (int i = 0; i < pb->length; i++) {
    printf("%d,", pb->val[i]);
  }
  printf("\nPositions:\t\t");
  for (int i = 0; i < pb->length; i++) {
    printf("%d,", pb->pos[i]);
  }
}

void display_area(Area *area) {
  printf("Virtual space values:\t\n");
  for (int i = 0; i < area->blockNum; i++) {
    printf("block%d--", i);
    for (int j = 0; j < area->blocks[i]->length; j++) {
      printf("%d,", area->blocks[i]->val[j]);
    }
    printf("\t");
    printf("\n");
  }
  printf("\nPositions:\t\t\n");
  for (int i = 0; i < area->blockNum; i++) {
    printf("block%d--", i);
    for (int j = 0; j < area->blocks[i]->length; j++) {
      printf("%d,", area->blocks[i]->pos[j]);
    }
    printf("\t");
    printf("\n");
  }
}

void display_bindex(BinDex *bindex, CODE *raw_data) {
  for (int i = 0; i < K; i++) {
    printf("Area%d:\n", i);
    display_area(bindex->areas[i]);
    printf("\t\n");
  }
  printf("Length:%d\t raw_data:", bindex->length);
  for (int i = 0; i < bindex->length; i++) {
    printf("%d,", raw_data[i]);
    if ((i + 1) % 4 == 0) printf("|");
    if ((i + 1) % 32 == 0) printf("--------");
  }
  printf("\t\n==\n");
  for (int i = 0; i < K - 1; i++) {
    printf("filterVector%d:\n", i);
    display_bitmap(bindex->filterVectors[i], bits_num_needed(bindex->length));
    printf("\t\n");
  }
}

void set_fv_val_less(BITS *bitmap, const CODE *val, CODE compare, POSTYPE n) {
  // Set values for filter vectors
  int i;
  for (i = 0; i + BITSWIDTH < (int)n; i += BITSWIDTH) {
    bitmap[i / BITSWIDTH] = gen_less_bits(val + i, compare, BITSWIDTH);
  }
  bitmap[i / BITSWIDTH] = gen_less_bits(val + i, compare, n - i);
}

int padding_fv_val_less(BITS *bitmap, POSTYPE length_old, const CODE *val_new, CODE compare, POSTYPE n) {
  // Padding new bit results to old bitmap to fill up a BITS variable, return
  // the number of bits needed for fill up a BITS variable

  if (length_old % BITSWIDTH == 0) return 0;  // No padding is needed for fill up a BITS

  int result_bits_count;
  int bitmap_insert_pos = length_old / BITSWIDTH;
  int padding = (bitmap_insert_pos + 1) * BITSWIDTH - length_old;
  result_bits_count = padding;
  if (padding > (int)n) {
    result_bits_count = n;  // Too little new data, still not filled up,
    // return the number of actually appended bits
  }
  for (int i = 0; i < result_bits_count; i++) {
    if (val_new[i] < compare) {
      bitmap[bitmap_insert_pos] += (1 << (padding - i - 1));
    }
  }
  return result_bits_count;
}

void append_fv_val_less(BITS *bitmap, POSTYPE length_old, const CODE *val, CODE compare, POSTYPE n) {
  int padding = padding_fv_val_less(bitmap, length_old, val, compare, n);
  int bitmap_insert_pos = (length_old - 1) / BITSWIDTH + 1;
  set_fv_val_less(bitmap + bitmap_insert_pos, val + padding, compare, n - padding);
}

inline POSTYPE num_insert_to_area(POSTYPE *areaStartIdx, int k, int n) {
  if (k < K - 1) {
    return areaStartIdx[k + 1] - areaStartIdx[k];
  } else {
    return n - areaStartIdx[k];
  }
}

CODE *data_sorted;
void init_bindex(BinDex *bindex, CODE *data, POSTYPE n) {
  bindex->length = n;
  POSTYPE avgAreaSize = n / K;

  CODE areaStartValues[K];
  POSTYPE areaStartIdx[K];

  data_sorted = (CODE *)malloc(n * sizeof(CODE));  // Sorted codes

  POSTYPE *pos = argsort(data, n);
  for (int i = 0; i < n; i++) {
    data_sorted[i] = data[pos[i]];
  }

  areaStartValues[0] = data_sorted[0];
  areaStartIdx[0] = 0;

  for (int i = 1; i < K; i++) {
    areaStartValues[i] = data_sorted[i * avgAreaSize];
    int j = i * avgAreaSize;
    if (areaStartValues[i] == areaStartValues[i - 1]) {
      areaStartIdx[i] = j;
    } else {
      // To find the first element which is less than startValue
      // TODO: in current implementation, an area must contain at least two
      // different values, area containing unique code should be considered in
      // future implementation
      while (data_sorted[j] == areaStartValues[i]) {
        j--;
      }
      areaStartIdx[i] = j + 1;
    }
  }
  // Build the areas
  std::thread threads[THREAD_NUM];
  for (int k = 0; k * THREAD_NUM < K; k++) {
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < K; j++) {
      int area_idx = k * THREAD_NUM + j;
      bindex->areas[area_idx] = (Area *)malloc(sizeof(Area));
      POSTYPE area_size = num_insert_to_area(areaStartIdx, area_idx, n);
      bindex->area_counts[area_idx] = area_size;
      threads[j] = std::thread(init_area, bindex->areas[area_idx], data_sorted + areaStartIdx[area_idx],
                               pos + areaStartIdx[area_idx], area_size);
    }
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < K; j++) {
      threads[j].join();
    }
  }

  // Accumulative adding
  for (int i = 1; i < K; i++) {
    bindex->area_counts[i] += bindex->area_counts[i - 1];
  }
  assert(bindex->area_counts[K - 1] == bindex->length);


  // Build the filterVectors
  for (int k = 0; k * THREAD_NUM < K - 1; k++) {
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      // Malloc 2 times of space, prepared for future appending
      bindex->filterVectors[k * THREAD_NUM + j] =
          (BITS *)aligned_alloc(SIMD_ALIGEN, 2 * bits_num_needed(n) * sizeof(BITS));
      threads[j] = std::thread(set_fv_val_less, bindex->filterVectors[k * THREAD_NUM + j], data,
                               area_start_value(bindex->areas[k * THREAD_NUM + j + 1]), n);
    }
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      threads[j].join();
    }
  }

  free(pos);
  free(data_sorted);
}

void append_to_bindex(BinDex *bindex, CODE *new_data, POSTYPE n, CODE *raw_data) {
  if (DEBUG_TIME_COUNT) timer.commonGetStartTime(1);

  POSTYPE *idx = argsort(new_data, n);
  CODE *data_sorted = (CODE *)malloc(n * sizeof(CODE));
  POSTYPE areaStartIdx[K];
  POSTYPE *new_pos = (POSTYPE *)malloc(n * sizeof(POSTYPE));
  areaStartIdx[0] = 0;
  int k = 1;
  for (POSTYPE i = 0; i < n; i++) {
    // TODO: multithread appending should(?) be done in future implementation
    data_sorted[i] = new_data[idx[i]];
    new_pos[i] = idx[i] + bindex->length;
    raw_data[bindex->length + i] = new_data[i];
    while (k < K && data_sorted[i] >= area_start_value(bindex->areas[k])) {
      // TODO: Here a naive linear search is used for calculating the
      // areaStartIdx, optimizations may(?may not) be done in future implementation
      areaStartIdx[k++] = i;
    }
  }
  while (k < K) {
    areaStartIdx[k++] = n;
  }
  std::thread threads[THREAD_NUM];
  int i, j;

  // Update the areas
  int accum_add_count = 0;
  for (k = 0; k * THREAD_NUM < K; k++) {
    if (DEBUG_TIME_COUNT) timer.commonGetStartTime(4);
    for (j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < K; j++) {
      i = k * THREAD_NUM + j;
      int num_added = num_insert_to_area(areaStartIdx, i, n);
      accum_add_count += num_added;
      threads[j] = std::thread(insert_to_area, bindex->areas[i], data_sorted + areaStartIdx[i],
                               new_pos + areaStartIdx[i], num_added, raw_data);
      bindex->area_counts[i] += accum_add_count;
    }
    for (j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < K; j++) {
      threads[j].join();
    }
    if (DEBUG_TIME_COUNT) timer.commonGetEndTime(4);
  }

  // Append to the filter vectors
  for (k = 0; k * THREAD_NUM < (K - 1); k++) {
    if (DEBUG_TIME_COUNT) timer.commonGetStartTime(6);
    for (j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      i = k * THREAD_NUM + j;
      threads[j] = std::thread(append_fv_val_less, bindex->filterVectors[i], bindex->length, new_data,
                               area_start_value(bindex->areas[i + 1]), n);
    }
    for (j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      threads[j].join();
    }
    if (DEBUG_TIME_COUNT) timer.commonGetEndTime(6);
  }

  bindex->length += n;

  free(idx);
  if (DEBUG_TIME_COUNT) timer.commonGetEndTime(1);
}

char *bin_repr(BITS x) {
  // Generate binary representation of a BITS variable
  int len = BITSWIDTH + BITSWIDTH / 4 + 1;
  char *result = (char *)malloc(sizeof(char) * len);
  BITS ref;
  int j = 0;
  for (int i = 0; i < BITSWIDTH; i++) {
    ref = 1 << (BITSWIDTH - i - 1);
    result[i + j] = (ref & x) ? '1' : '0';
    if ((i + 1) % 4 == 0) {
      j++;
      result[i + j] = '-';
    }
  }
  result[len - 1] = '\0';
  return result;
}

void display_bitmap(BITS *bitmap, int bitmap_len) {
  for (int i = 0; i < bitmap_len; i++) {
    printf("%s;", bin_repr(*(bitmap + i)));
  }
}

void copy_bitmap(BITS *result, BITS *ref, int n, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  memcpy(result, ref, n * sizeof(BITS));

  // for (int i = 0; i < n; i++) {
  //     result[i] = ref[i];
  // }
}

void copy_bitmap_not(BITS *result, BITS *ref, int start_n, int end_n, int t_id) {
  int jobs = ROUNDUP_DIVIDE(end_n - start_n, THREAD_NUM);
  int start = start_n + t_id * jobs;
  int end = start_n + (t_id + 1) * jobs;
  if (end > end_n) end = end_n;
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  // memcpy(result, ref, n * sizeof(BITS));

  // TODO: bitwise operation on large memory block?
  for (int i = start; i < end; i++) {
    result[i] = ~(ref[i]);
  }
}

void copy_bitmap_bt(BITS *result, BITS *ref_l, BITS *ref_r, int start_n, int end_n, int t_id) {
  int jobs = ROUNDUP_DIVIDE(end_n - start_n, THREAD_NUM);
  int start = start_n + t_id * jobs;
  int end = start_n + (t_id + 1) * jobs;
  if (end > end_n) end = end_n;

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  // memcpy(result, ref, n * sizeof(BITS));

  // TODO: bitwise operation on large memory block?
  for (int i = start; i < end; i++) {
    result[i] = ref_r[i] & (~(ref_l[i]));
  }
}

void copy_bitmap_bt_simd(BITS *to, BITS *from_l, BITS *from_r, int bitmap_len, int t_id) {
  int jobs = ((bitmap_len / SIMD_JOB_UNIT - 1) / THREAD_NUM + 1) * SIMD_JOB_UNIT;

  assert(jobs % SIMD_JOB_UNIT == 0);
  assert(bitmap_len % SIMD_JOB_UNIT == 0);
  assert(jobs * THREAD_NUM >= bitmap_len);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }

  int cur = t_id * jobs;
  int end = (t_id + 1) * jobs;
  if (end > bitmap_len) end = bitmap_len;

  while (cur < end) {
    __m256i buf_l = _mm256_load_si256((__m256i *)(from_l + cur));
    __m256i buf_r = _mm256_load_si256((__m256i *)(from_r + cur));
    __m256i buf = _mm256_andnot_si256(buf_l, buf_r);
    _mm256_store_si256((__m256i *)(to + cur), buf);
    cur += SIMD_JOB_UNIT;
  }
}

void copy_bitmap_simd(BITS *to, BITS *from, int bitmap_len, int t_id) {
  int jobs = ((bitmap_len / SIMD_JOB_UNIT - 1) / THREAD_NUM + 1) * SIMD_JOB_UNIT;

  assert(jobs % SIMD_JOB_UNIT == 0);
  assert(bitmap_len % SIMD_JOB_UNIT == 0);
  assert(jobs * THREAD_NUM >= bitmap_len);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }

  int cur = t_id * jobs;
  int end = (t_id + 1) * jobs;
  if (end > bitmap_len) end = bitmap_len;

  while (cur < end) {
    __m256i buf = _mm256_load_si256((__m256i *)(from + cur));
    _mm256_store_si256((__m256i *)(to + cur), buf);
    cur += SIMD_JOB_UNIT;
  }
}

void copy_bitmap_not_simd(BITS *to, BITS *from, int bitmap_len, int t_id) {
  int jobs = ((bitmap_len / SIMD_JOB_UNIT - 1) / THREAD_NUM + 1) * SIMD_JOB_UNIT;

  assert(jobs % SIMD_JOB_UNIT == 0);
  assert(bitmap_len % SIMD_JOB_UNIT == 0);
  assert(jobs * THREAD_NUM >= bitmap_len);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }

  int cur = t_id * jobs;
  int end = (t_id + 1) * jobs;
  if (end > bitmap_len) end = bitmap_len;

  while (cur < end) {
    __m256i buf = _mm256_load_si256((__m256i *)(from + cur));
    buf = ~buf;
    _mm256_store_si256((__m256i *)(to + cur), buf);
    cur += SIMD_JOB_UNIT;
  }
}

void memset_numa0(BITS *p, int val, int n, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  int avg_workload = (n / (THREAD_NUM * SIMD_ALIGEN)) * SIMD_ALIGEN;
  int start = t_id * avg_workload;
  // int end = start + avg_workload;
  int end = t_id == (THREAD_NUM - 1) ? n : start + avg_workload;
  memset(p + start, val, (end - start) * sizeof(BITS));
  // if (t_id == THREAD_NUM) {
  //   BITS val_bits = 0U;
  //   for (int i = 0; i < sizeof(BITS); i++) {
  //     val_bits |= ((unsigned int)val) << (8 * i);
  //   }
  //   for (; end < n; end++) {
  //     p[end] = val_bits;
  //   }
  // }
}

void memset_mt(BITS *p, int val, int n) {
  std::thread threads[THREAD_NUM];
  for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
    threads[t_id] = std::thread(memset_numa0, p, val, n, t_id);
  }
  for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
    threads[t_id].join();
  }
}

void copy_filter_vector(BinDex *bindex, BITS *result, int k) {
  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);
  // BITS* result = (BITS*)aligned_alloc(SIMD_ALIGEN, bitmap_len *
  // sizeof(BITS));

  if (k < 0) {
    memset_mt(result, 0, bitmap_len);
    return;
  }

  if (k >= (K - 1)) {
    memset_mt(result, 0xFF, bitmap_len);  // Slower(?) than a loop
    return;
  }

  // simd copy
  // int mt_bitmap_n = (bitmap_len / SIMD_JOB_UNIT) * SIMD_JOB_UNIT; // must
  // be SIMD_JOB_UNIT aligened for (int i = 0; i < THREAD_NUM; i++)
  //     threads[i] = std::thread(copy_bitmap_simd, result,
  //     bindex->filterVectors[k], mt_bitmap_n, i);
  // for (int i = 0; i < THREAD_NUM; i++)
  //     threads[i].join();
  // memcpy(result + mt_bitmap_n, bindex->filterVectors[k] + mt_bitmap_n,
  // bitmap_len - mt_bitmap_n);

  // naive copy
  int avg_workload = bitmap_len / THREAD_NUM;
  int i;
  for (i = 0; i < THREAD_NUM - 1; i++) {
    threads[i] = std::thread(copy_bitmap, result + (i * avg_workload), bindex->filterVectors[k] + (i * avg_workload),
                             avg_workload, i);
  }
  threads[i] = std::thread(copy_bitmap, result + (i * avg_workload), bindex->filterVectors[k] + (i * avg_workload),
                           bitmap_len - (i * avg_workload), i);

  for (i = 0; i < THREAD_NUM; i++) {
    threads[i].join();
  }
}

void copy_filter_vector_not(BinDex *bindex, BITS *result, int k) {
  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);
  // BITS* result = (BITS*)aligned_alloc(SIMD_ALIGEN, bitmap_len *
  // sizeof(BITS));

  if (k < 0) {
    memset_mt(result, 0xFF, bitmap_len);
    return;
  }

  if (k >= (K - 1)) {
    memset_mt(result, 0, bitmap_len);  // Slower(?) than a loop
    return;
  }

  // simd copy not
  int mt_bitmap_n = (bitmap_len / SIMD_JOB_UNIT) * SIMD_JOB_UNIT;  // must be SIMD_JOB_UNIT aligened
  for (int i = 0; i < THREAD_NUM; i++)
    threads[i] = std::thread(copy_bitmap_not_simd, result, bindex->filterVectors[k], mt_bitmap_n, i);
  for (int i = 0; i < THREAD_NUM; i++) threads[i].join();
  for (int i = 0; i < bitmap_len - mt_bitmap_n; i++) {
    (result + mt_bitmap_n)[i] = ~((bindex->filterVectors[k] + mt_bitmap_n)[i]);
  }

  // naive copy
  // for (int i = 0; i < THREAD_NUM; i++) {
  //   threads[i] = std::thread(copy_bitmap_not, result,
  //   bindex->filterVectors[k],
  //                            0, bitmap_len, i);
  // }

  // for (int i = 0; i < THREAD_NUM; i++) {
  //   threads[i].join();
  // }

  return;
}

void copy_filter_vector_bt(BinDex *bindex, BITS *result, int kl, int kr) {
  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);

  // TODO: finish this
  if (kr < 0) {
    // assert(0);
    // printf("1\n");
    memset_mt(result, 0, bitmap_len);
    return;
  } else if (kr >= (K - 1)) {
    // assert(0);
    // printf("2\n");
    copy_filter_vector_not(bindex, result, kl);
    return;
  }
  if (kl < 0) {
    // assert(0);
    // printf("3\n");
    copy_filter_vector(bindex, result, kr);
    return;
  } else if (kl >= (K - 1)) {
    // assert(0);
    // printf("4\n");
    memset_mt(result, 0, bitmap_len);  // Slower(?) than a loop
    return;
  }

  // simd copy_bt
  int mt_bitmap_n = (bitmap_len / SIMD_JOB_UNIT) * SIMD_JOB_UNIT;  // must be SIMD_JOB_UNIT aligened
  for (int i = 0; i < THREAD_NUM; i++)
    threads[i] =
        std::thread(copy_bitmap_bt_simd, result, bindex->filterVectors[kl], bindex->filterVectors[kr], mt_bitmap_n, i);
  for (int i = 0; i < THREAD_NUM; i++) threads[i].join();
  for (int i = 0; i < bitmap_len - mt_bitmap_n; i++) {
    (result + mt_bitmap_n)[i] =
        (~((bindex->filterVectors[kl] + mt_bitmap_n)[i])) & ((bindex->filterVectors[kr] + mt_bitmap_n)[i]);
  }

  // naive copy
  // for (int i = 0; i < THREAD_NUM; i++) {
  //   threads[i] = std::thread(copy_bitmap_bt, result,
  //   bindex->filterVectors[kl],
  //                            bindex->filterVectors[kr], 0, bitmap_len, i);
  // }

  // for (int i = 0; i < THREAD_NUM; i++) {
  //   threads[i].join();
  // }
}

void copy_bitmap_xor_simd(BITS *to, BITS *bitmap1, BITS *bitmap2,
                          int bitmap_len, int t_id) {
  int jobs =
      ((bitmap_len / SIMD_JOB_UNIT - 1) / THREAD_NUM + 1) * SIMD_JOB_UNIT;

  assert(jobs % SIMD_JOB_UNIT == 0);
  assert(bitmap_len % SIMD_JOB_UNIT == 0);
  assert(jobs * THREAD_NUM >= bitmap_len);

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }

  int cur = t_id * jobs;
  int end = (t_id + 1) * jobs;
  if (end > bitmap_len) end = bitmap_len;

  while (cur < end) {
    __m256i buf1 = _mm256_load_si256((__m256i *)(bitmap1 + cur));
    __m256i buf2 = _mm256_load_si256((__m256i *)(bitmap2 + cur));
    __m256i buf = _mm256_andnot_si256(buf1, buf2);
    _mm256_store_si256((__m256i *)(to + cur), buf);
    cur += SIMD_JOB_UNIT;
  }
}

void copy_filter_vector_xor(BinDex *bindex, BITS *result, int kl, int kr) {
  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);

  // TODO: finish this
  if (kr < 0) {
    assert(0);
    // printf("1\n");
    memset_mt(result, 0, bitmap_len);
    return;
  } else if (kr >= (K - 1)) {
    assert(0);
    // printf("2\n");
    copy_filter_vector_not(bindex, result, kl);
    return;
  }
  if (kl < 0) {
    // assert(0);
    // printf("3\n");
    copy_filter_vector(bindex, result, kr);
    return;
  } else if (kl >= (K - 1)) {
    assert(0);
    // printf("4\n");
    memset_mt(result, 0, bitmap_len);  // Slower(?) than a loop
    return;
  }

  // simd copy_xor
  int mt_bitmap_n = (bitmap_len / SIMD_JOB_UNIT) *
                    SIMD_JOB_UNIT;  // must be SIMD_JOB_UNIT aligened
  for (int i = 0; i < THREAD_NUM; i++)
    threads[i] =
        std::thread(copy_bitmap_xor_simd, result, bindex->filterVectors[kl],
                    bindex->filterVectors[kr], mt_bitmap_n, i);
  for (int i = 0; i < THREAD_NUM; i++) threads[i].join();
  for (int i = 0; i < bitmap_len - mt_bitmap_n; i++) {
    (result + mt_bitmap_n)[i] = ((bindex->filterVectors[kl] + mt_bitmap_n)[i]) ^
                                ((bindex->filterVectors[kr] + mt_bitmap_n)[i]);
  }

  // naive copy
  // for (int i = 0; i < THREAD_NUM; i++) {
  //   threads[i] = std::thread(copy_bitmap_bt, result,
  //   hydex->filterVectors[kl],
  //                            hydex->filterVectors[kr], 0, bitmap_len, i);
  // }

  // for (int i = 0; i < THREAD_NUM; i++) {
  //   threads[i].join();
  // }
}

int in_which_area(BinDex *bindex, CODE compare) {
  // TODO: could return -1?
  // Return the last area whose startValue is less than 'compare'
  // Obviously here a naive linear search is enough
  // Return -1 if 'compare' less than the first value in the virtual space
  // TODO: outdated comments, bad code, but worked
  if (compare < area_start_value(bindex->areas[0])) return -1;
  for (int i = 0; i < K; i++) {
    CODE area_sv = area_start_value(bindex->areas[i]);
    if (area_sv == compare) return i;
    if (area_sv > compare) return i - 1;
  }
  return K - 1;
}

int in_which_block(Area *area, CODE compare) {
  // Binary search here to find which block the value of compare should locate
  // in (i.e. the last block whose startValue is less than 'compare')
  // TODO: outdated comments, bad code, but worked
  assert(compare >= area_start_value(area));
  int res = area->blockNum - 1;
  for (int i = 0; i < area->blockNum; i++) {
    CODE area_sv = block_start_value(area->blocks[i]);
    if (area_sv == compare) {
      res = i;
      break;
    }
    if (area_sv > compare) {
      res = i - 1;
      break;
    }
  }
  if (res) {
    pos_block *pre_blk = area->blocks[res - 1];
    if (pre_blk->val[pre_blk->length - 1] == compare) {
      res--;
    }
  }
  return res;

  int low = 0, high = area->blockNum, mid = (low + high) / 2;
  while (low < high) {
    if (compare <= block_start_value(area->blocks[mid])) {
      high = mid;
    } else {
      low = mid + 1;
    }
    mid = (low + high) / 2;
  }
  return mid - 1;
  /*
  // TODO: do we need this?
  int res = mid - 1;
  if (res) {
    pos_block *pre_blk = area->blocks[res - 1];
    if (pre_blk->val[pre_blk->length - 1] == compare) {
      res--;
    }
  }
  return res;
  */
}

int on_which_pos(pos_block *pb, CODE compare) {
  // Find the first value which is no less than 'compare', return pb->length if
  // all data in the block are less than compare
  assert(compare >= pb->val[0]);
  int low = 0, high = pb->length, mid = (low + high) / 2;
  while (low < high) {
    if (pb->val[mid] >= compare) {
      high = mid;
    } else {
      low = mid + 1;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

inline void refine(BITS *bitmap, POSTYPE pos) { bitmap[pos >> BITSSHIFT] ^= (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)); }

void refine_positions(BITS *bitmap, POSTYPE *pos, POSTYPE n) {
  for (int i = 0; i < n; i++) {
    refine(bitmap, *(pos + i));
  }
}

void refine_positions_mt(BITS *bitmap, Area *area, int start_blk_idx, int end_blk_idx, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  int jobs = ROUNDUP_DIVIDE(end_blk_idx - start_blk_idx, THREAD_NUM);
  int cur = start_blk_idx + t_id * jobs;
  int end = start_blk_idx + (t_id + 1) * jobs;
  if (end > end_blk_idx) end = end_blk_idx;

  // int prefetch_stride = 6;
  while (cur < end) {
    POSTYPE *pos_list = area->blocks[cur]->pos;
    POSTYPE n = area->blocks[cur]->length;
    int i;
    for (i = 0; i + prefetch_stride < n; i++) {
      if(prefetch_stride) {
        __builtin_prefetch(&bitmap[*(pos_list + i + prefetch_stride) >> BITSSHIFT], 1, 1);
      }
      POSTYPE pos = *(pos_list + i);
      __sync_fetch_and_xor(&bitmap[pos >> BITSSHIFT], (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)));
    }
    while (i < n) {
      POSTYPE pos = *(pos_list + i);
      __sync_fetch_and_xor(&bitmap[pos >> BITSSHIFT], (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)));
      i++;
    }
    cur++;
  }
}


void refine_result_bitmap(BITS *bitmap_a, BITS *bitmap_b, int start_idx, int end_idx, int t_id) {

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  // int prefetch_stride = 6;
  int i;
  for (i = start_idx; i + prefetch_stride < end_idx; i++) {
    /* if(prefetch_stride) {
      __builtin_prefetch(&bitmap_a[*(pos_list + i + prefetch_stride) >> BITSSHIFT], 1, 1);
    } */
    __sync_fetch_and_and(&bitmap_a[i], bitmap_b[i]);
  }

  while (i < end_idx) {
    __sync_fetch_and_and(&bitmap_a[i], bitmap_b[i]);
    i++;
  }

}

void xor_bitmap_mt(BITS *bitmap, BITS *bitmap1, BITS *bitmap2, int start_n, int end_n, int t_id) {
  int jobs = ROUNDUP_DIVIDE(end_n - start_n, THREAD_NUM);
  int start = start_n + t_id * jobs;
  int end = start_n + (t_id + 1) * jobs;
  if (end > end_n) end = end_n;

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }

  // TODO: bitwise operation on large memory block?
  for (int i = start; i < end; i++) {
    bitmap[i] = bitmap1[i] ^ bitmap2[i];
  }
}

void set_eq_bitmap_mt(BITS *bitmap, Area *area, CODE compare, int start_blk_idx, int end_blk_idx, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }

  int jobs = ROUNDUP_DIVIDE(end_blk_idx - start_blk_idx, THREAD_NUM);
  int start = start_blk_idx + t_id * jobs;
  int end = start_blk_idx + (t_id + 1) * jobs;
  if (end > end_blk_idx) end = end_blk_idx;

  // TODO: prefetch?
  // TODO: use direct pos_idx
  for (int i = start; i < end && block_start_value(area->blocks[i]) <= compare; i++) {
    pos_block *blk = area->blocks[i];
    POSTYPE *pos_list = blk->pos;
    POSTYPE n = blk->length;
    for (int j = 0; j < n && blk->val[j] <= compare; j++) {
      if (blk->val[j] == compare) {
        POSTYPE pos = *(pos_list + j);
        __sync_fetch_and_xor(&bitmap[pos >> BITSSHIFT], (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)));
      }
    }
  }
}

void refine_positions_in_blks_mt(BITS *bitmap, Area *area, int start_blk_idx,
                                 int end_blk_idx, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  int jobs = ROUNDUP_DIVIDE(end_blk_idx - start_blk_idx, THREAD_NUM);
  int cur = start_blk_idx + t_id * jobs;
  int end = start_blk_idx + (t_id + 1) * jobs;
  if (end > end_blk_idx) end = end_blk_idx;

  int prefetch_stride = 6;
  while (cur < end) {
    POSTYPE *pos_list = area->blocks[cur]->pos;
    POSTYPE n = area->blocks[cur]->length;
    int i;
    for (i = 0; i + prefetch_stride < n; i++) {
      __builtin_prefetch(
          &bitmap[*(pos_list + i + prefetch_stride) >> BITSSHIFT], 1, 1);
      POSTYPE pos = *(pos_list + i);
      __sync_fetch_and_xor(&bitmap[pos >> BITSSHIFT],
                           (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)));
    }
    while (i < n) {
      POSTYPE pos = *(pos_list + i);
      __sync_fetch_and_xor(&bitmap[pos >> BITSSHIFT],
                           (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)));
      i++;
    }
    cur++;
  }
}

void bindex_scan_lt(BinDex *bindex, BITS *result, CODE compare) {
  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);
  int area_idx = in_which_area(bindex, compare);
  if (area_idx < 0) {
    // 'compare' less than all raw_data, return all zero result
    // BITS* result = (BITS*)malloc(sizeof(BITS) * bitmap_len);
    memset_mt(result, 0, bitmap_len);
    return;
  }
  Area *area = bindex->areas[area_idx];
  int block_idx = in_which_block(area, compare);
  int pos_idx = on_which_pos(area->blocks[block_idx], compare);
  // Do an estimation to select the filter vector which is most
  // similar to the correct result
  int is_upper_fv;
  int start_blk_idx, end_blk_idx;
  if (block_idx < bindex->areas[area_idx]->blockNum / 2) {
    is_upper_fv = 1;
    start_blk_idx = 0;
    end_blk_idx = block_idx;
  } else {
    is_upper_fv = 0;
    start_blk_idx = block_idx + 1;
    end_blk_idx = area->blockNum;
  }

  // fprintf(stderr, "area_idx: %d\nblock_idx: %d\npos_idx: %d\nis_upper_fv:
  // %d\nstart_blk_idx: %d\nend_blk_idx: %d\n", area_idx, block_idx, pos_idx,
  // is_upper_fv, start_blk_idx, end_blk_idx);

  // clang-format off
  PRINT_EXCECUTION_TIME("copy",
                        copy_filter_vector(bindex,
                                            result,
                                            is_upper_fv ? (area_idx - 1) : (area_idx)))

  PRINT_EXCECUTION_TIME("refine",
                        for (int i = 0; i < THREAD_NUM; i++) {
                          threads[i] = std::thread(refine_positions_mt, result, area, start_blk_idx, end_blk_idx, i);
                        }

                        for (int i = 0; i < THREAD_NUM; i++) {
                          threads[i].join();
                        }

                        if (is_upper_fv) {
                          refine_positions(result, area->blocks[block_idx]->pos, pos_idx);
                        } else {
                          refine_positions(result, area->blocks[block_idx]->pos + pos_idx, area->blocks[block_idx]->length - pos_idx);
                        })
  // clang-format on
}

void bindex_scan_le(BinDex *bindex, BITS *result, CODE compare) {
  // TODO: (compare + 1) overflow
  bindex_scan_lt(bindex, result, compare + 1);
}

void bindex_scan_gt(BinDex *bindex, BITS *result, CODE compare) {
  // TODO: (compare + 1) overflow
  compare = compare + 1;

  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);
  int area_idx = in_which_area(bindex, compare);
  if (area_idx < 0) {
    // 'compare' less than all raw_data, return all 1 result
    // BITS* result = (BITS*)malloc(sizeof(BITS) * bitmap_len);
    memset_mt(result, 0xFF, bitmap_len);
    return;
  }
  Area *area = bindex->areas[area_idx];
  int block_idx = in_which_block(area, compare);
  int pos_idx = on_which_pos(area->blocks[block_idx], compare);

  // Do an estimation to select the filter vector which is most
  // similar to the correct result
  int is_upper_fv;
  int start_blk_idx, end_blk_idx;
  if (block_idx < bindex->areas[area_idx]->blockNum / 2) {
    is_upper_fv = 1;
    start_blk_idx = 0;
    end_blk_idx = block_idx;
  } else {
    is_upper_fv = 0;
    start_blk_idx = block_idx + 1;
    end_blk_idx = area->blockNum;
  }

  // clang-format off
  PRINT_EXCECUTION_TIME("copy",
                        copy_filter_vector_not(bindex,
                                                result,
                                                is_upper_fv ? (area_idx - 1) : (area_idx)))

  PRINT_EXCECUTION_TIME("refine",
                        for (int i = 0; i < THREAD_NUM; i++) {
                          threads[i] = std::thread(refine_positions_mt, result, area, start_blk_idx, end_blk_idx, i);
                        }

                        for (int i = 0; i < THREAD_NUM; i++) {
                          threads[i].join();
                        }

                        if (is_upper_fv) {
                          refine_positions(result, area->blocks[block_idx]->pos, pos_idx);
                        } else {
                          refine_positions(result, area->blocks[block_idx]->pos + pos_idx, area->blocks[block_idx]->length - pos_idx);
                        })
  // clang-format on
}

void bindex_scan_ge(BinDex *bindex, BITS *result, CODE compare) {
  // TODO: (compare - 1) overflow
  bindex_scan_gt(bindex, result, compare - 1);
}

void bindex_scan_bt(BinDex *bindex, BITS *result, CODE compare1, CODE compare2) {
  assert(compare2 > compare1);
  // TODO: (compare1 + 1) overflow
  compare1 = compare1 + 1;

  std::thread threads[THREAD_NUM];
  int bitmap_len = bits_num_needed(bindex->length);

  // x > compare1
  int area_idx_l = in_which_area(bindex, compare1);
  if (area_idx_l < 0) {
    // TODO: finish this
    // assert(0);
    // 'compare' less than all raw_data, return all 1 result
    // BITS* result = (BITS*)malloc(sizeof(BITS) * bitmap_len);
    memset_mt(result, 0xFF, bitmap_len);
    return;
  }
  Area *area_l = bindex->areas[area_idx_l];
  int block_idx_l = in_which_block(area_l, compare1);
  int pos_idx_l = on_which_pos(area_l->blocks[block_idx_l], compare1);

  // Do an estimation to select the filter vector which is most
  // similar to the correct result
  int is_upper_fv_l;
  int start_blk_idx_l, end_blk_idx_l;
  if (block_idx_l < bindex->areas[area_idx_l]->blockNum / 2) {
    is_upper_fv_l = 1;
    start_blk_idx_l = 0;
    end_blk_idx_l = block_idx_l;
  } else {
    is_upper_fv_l = 0;
    start_blk_idx_l = block_idx_l + 1;
    end_blk_idx_l = area_l->blockNum;
  }

  // x < compare2
  int area_idx_r = in_which_area(bindex, compare2);
  if (area_idx_r < 0) {
    // TODO: finish this
    assert(0);
    // 'compare' less than all raw_data, return all zero result
    // BITS* result = (BITS*)malloc(sizeof(BITS) * bitmap_len);
    memset_mt(result, 0, bitmap_len);
    return;
  }
  Area *area_r = bindex->areas[area_idx_r];
  int block_idx_r = in_which_block(area_r, compare2);
  int pos_idx_r = on_which_pos(area_r->blocks[block_idx_r], compare2);
  // Do an estimation to select the filter vector which is most
  // similar to the correct result
  int is_upper_fv_r;
  int start_blk_idx_r, end_blk_idx_r;
  if (block_idx_r < bindex->areas[area_idx_r]->blockNum / 2) {
    is_upper_fv_r = 1;
    start_blk_idx_r = 0;
    end_blk_idx_r = block_idx_r;
  } else {
    is_upper_fv_r = 0;
    start_blk_idx_r = block_idx_r + 1;
    end_blk_idx_r = area_r->blockNum;
  }

  // clang-format off
  PRINT_EXCECUTION_TIME("copy",
                        copy_filter_vector_bt(bindex,
                                              result,
                                              is_upper_fv_l ? (area_idx_l - 1) : (area_idx_l),
                                              is_upper_fv_r ? (area_idx_r - 1) : (area_idx_r))
                        )

  PRINT_EXCECUTION_TIME("refine",
                        // refine left part
                        for (int i = 0; i < THREAD_NUM; i++)
                          threads[i] = std::thread(refine_positions_mt, result, area_l, start_blk_idx_l, end_blk_idx_l, i);
                        for (int i = 0; i < THREAD_NUM; i++) threads[i].join();

                        if (is_upper_fv_l)
                          refine_positions(result, area_l->blocks[block_idx_l]->pos, pos_idx_l);
                        else
                          refine_positions(result, area_l->blocks[block_idx_l]->pos + pos_idx_l, area_l->blocks[block_idx_l]->length - pos_idx_l);

                        // refine right part
                        for (int i = 0; i < THREAD_NUM; i++)
                          threads[i] = std::thread(refine_positions_mt, result, area_r, start_blk_idx_r, end_blk_idx_r, i);
                        for (int i = 0; i < THREAD_NUM; i++) threads[i].join();

                        if (is_upper_fv_r)
                          refine_positions(result, area_r->blocks[block_idx_r]->pos, pos_idx_r);
                        else
                          refine_positions(result, area_r->blocks[block_idx_r]->pos + pos_idx_r, area_r->blocks[block_idx_r]->length - pos_idx_r);
                        )
  // clang-format on
}

void bindex_scan_eq(BinDex *bindex, BITS *result, CODE compare) {
  int bitmap_len = bits_num_needed(bindex->length);
  std::thread threads[THREAD_NUM];

  int area_idx = in_which_area(bindex, compare);
  assert(area_idx >= 0 && area_idx <= K - 1);

  if (area_idx != K - 1 &&
      area_start_value(bindex->areas[area_idx + 1]) == compare) {
    // nm > N / K
    // result1 = hydex_scan_lt(compare)
    // result2 = hydex_scan_lt(compare + 1)
    // result = result1 ^ result2
    // TODO: (compare1 + 1) overflow
    std::thread threads[THREAD_NUM];
    int bitmap_len = bits_num_needed(bindex->length);

    // compare
    int area_idx = in_which_area(bindex, compare);
    if (area_idx < 0) {
      bindex_scan_lt(bindex, result2, compare + 1);
      return;
    }
    Area *area = bindex->areas[area_idx];
    int block_idx = in_which_block(area, compare);
    int pos_idx = on_which_pos(area->blocks[block_idx], compare);
    // Do an estimation to select the filter vector which is most
    // similar to the correct result
    int is_upper_fv;
    int start_blk_idx, end_blk_idx;
    if (block_idx < bindex->areas[area_idx]->blockNum / 2) {
      is_upper_fv = 1;
      start_blk_idx = 0;
      end_blk_idx = block_idx;
    } else {
      is_upper_fv = 0;
      start_blk_idx = block_idx + 1;
      end_blk_idx = area->blockNum;
    }

    // compare + 1
    CODE compare1 = compare + 1;
    int area_idx1 = in_which_area(bindex, compare1);
    if (area_idx1 < 0) {
      assert(0);
      return;
    }
    Area *area1 = bindex->areas[area_idx1];
    int block_idx1 = in_which_block(area1, compare1);
    int pos_idx1 = on_which_pos(area1->blocks[block_idx1], compare1);
    // Do an estimation to select the filter vector which is most
    // similar to the correct result
    int is_upper_fv1;
    int start_blk_idx1, end_blk_idx1;
    if (area_idx1 == 0 || (block_idx1 < bindex->areas[area_idx1]->blockNum / 2 &&
                           area_start_value(bindex->areas[area_idx1 - 1]) !=
                               area_start_value(area1))) {
      is_upper_fv1 = 1;
      start_blk_idx1 = 0;
      end_blk_idx1 = block_idx1;
    } else {
      is_upper_fv1 = 0;
      start_blk_idx1 = block_idx1 + 1;
      end_blk_idx1 = area1->blockNum;
    }

    // clang-format off
    PRINT_EXCECUTION_TIME("copy",
                        copy_filter_vector_xor(bindex,
                                              result,
                                              is_upper_fv ? (area_idx - 1) : (area_idx),
                                              is_upper_fv1 ? (area_idx1 - 1) : (area_idx1))
                        )

    PRINT_EXCECUTION_TIME("refine",
                        // refine left part
                        for (int i = 0; i < THREAD_NUM; i++)
                          threads[i] = std::thread(refine_positions_in_blks_mt, result, area, start_blk_idx, end_blk_idx, i);
                        for (int i = 0; i < THREAD_NUM; i++) threads[i].join();

                        if (is_upper_fv)
                          refine_positions(result, area->blocks[block_idx]->pos, pos_idx);
                        else
                          refine_positions(result, area->blocks[block_idx]->pos + pos_idx, area->blocks[block_idx]->length - pos_idx);

                        // refine right part
                        for (int i = 0; i < THREAD_NUM; i++)
                          threads[i] = std::thread(refine_positions_in_blks_mt, result, area1, start_blk_idx1, end_blk_idx1, i);
                        for (int i = 0; i < THREAD_NUM; i++) threads[i].join();

                        if (is_upper_fv1)
                          refine_positions(result, area1->blocks[block_idx1]->pos, pos_idx1);
                        else
                          refine_positions(result, area1->blocks[block_idx1]->pos + pos_idx1, area1->blocks[block_idx1]->length - pos_idx1);
                        )
    // clang-format on
  } else {
    // nm < N / K
    memset_mt(result, 0, bitmap_len);

    Area *area = bindex->areas[area_idx];
    int block_idx = in_which_block(area, compare);
    // int pos_idx = on_which_pos(area->blocks[block_idx], compare);

    for (int i = 0; i < THREAD_NUM; i++) {
      threads[i] = std::thread(set_eq_bitmap_mt, result, area, compare,
                               block_idx, area->blockNum, i);
    }

    for (int i = 0; i < THREAD_NUM; i++) {
      threads[i].join();
    }
  }
}

void check_worker(CODE *codes, int n, BITS *bitmap, CODE target1, CODE target2, OPERATOR OP, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  int avg_workload = n / THREAD_NUM;
  int start = t_id * avg_workload;
  int end = t_id == (THREAD_NUM - 1) ? n : start + avg_workload;
  for (int i = start; i < end; i++) {
    int data = codes[i];
    int truth;
    switch (OP) {
      case EQ:
        truth = data == target1;
        break;
      case LT:
        truth = data < target1;
        break;
      case GT:
        truth = data > target1;
        break;
      case LE:
        truth = data <= target1;
        break;
      case GE:
        truth = data >= target1;
        break;
      case BT:
        truth = data > target1 && data < target2;
        break;
      default:
        assert(0);
    }
    int res = !!(bitmap[i >> BITSSHIFT] & (1U << (BITSWIDTH - 1 - i % BITSWIDTH)));
    if (truth != res) {
      std::cerr << "raw data[" << i << "]: " << codes[i] << ", truth: " << truth << ", res: " << res << std::endl;
      assert(truth == res);
    }
  }
}

void check(BinDex *bindex, BITS *bitmap, CODE target1, CODE target2, OPERATOR OP, CODE *raw_data) {
  std::cout << "checking, target1: " << target1 << " target2: " << target2 << std::endl;
  std::thread threads[THREAD_NUM];
  for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
    threads[t_id] = std::thread(check_worker, raw_data, bindex->length, bitmap, target1, target2, OP, t_id);
  }
  for (int t_id = 0; t_id < THREAD_NUM; t_id++) {
    threads[t_id].join();
  }
  std::cout << "CHECK PASSED!" << std::endl;
}

void check_st(BinDex *bindex, BITS *bitmap, CODE target1, CODE target2, OPERATOR OP, CODE *raw_data) {
  printf("checking, target1: %u, target2: %u, OP: %d\n", target1, target2, OP);
  assert(OP != BT || target1 <= target2);
  for (int i = 0; i < bindex->length; i++) {
    int data = raw_data[i];
    int truth;
    switch (OP) {
      case EQ:
        truth = data == target1;
        break;
      case LT:
        truth = data < target1;
        break;
      case GT:
        truth = data > target1;
        break;
      case LE:
        truth = data <= target1;
        break;
      case GE:
        truth = data >= target1;
        break;
      case BT:
        truth = data > target1 && data < target2;
        break;
      default:
        assert(0);
    }
    int res = !!(bitmap[i >> BITSSHIFT] & (1U << (BITSWIDTH - 1 - i % BITSWIDTH)));
    if (truth != res) {
      fprintf(stderr, "raw_data[%d]: %d, truth: %d, res: %d\n", i, raw_data[i], truth, res);
      assert(truth == res);
      exit(-1);
    }
  }
  printf("CHECK PASS\n");
}

void free_pos_block(pos_block *pb) {
  free(pb->pos);
  free(pb->val);

  free(pb);
}

void free_area(Area *area) {
  for (int i = 0; i < area->blockNum; i++) {
    free_pos_block(area->blocks[i]);
  }

  free(area);
}

void free_bindex(BinDex *bindex, CODE *raw_data) {
  // CODE *raw_data
  free(raw_data);

  // BITS *filterVectors[K - 1]
  for (int i = 0; i < K - 1; i++) {
    free(bindex->filterVectors[i]);
  }

  // Area *areas[K]
  for (int i = 0; i < K; i++) {
    free_area(bindex->areas[i]);
  }

  free(bindex);
}

uint32_t str2uint32(const char *s) {
  uint32_t sum = 0;

  while (*s) {
    sum = sum * 10 + (*s++ - '0');
  }

  return sum;
}

std::vector<CODE> get_target_numbers(const char *s) {
  std::string input(s);
  std::stringstream ss(input);
  std::string value;
  std::vector<CODE> result;
  while (std::getline(ss, value, ',')) {
    result.push_back((CODE)stod(value));
    // result.push_back(str2uint32(value.c_str()));
  }
  return result;
}

std::vector<CODE> get_target_numbers(string s) {
  std::stringstream ss(s);
  std::string value;
  std::vector<CODE> result;
  while (std::getline(ss, value, ',')) {
    result.push_back((CODE)stod(value));
    // result.push_back(str2uint32(value.c_str()));
  }
  return result;
}

void raw_scan(BinDex *bindex, BITS *bitmap, CODE target1, CODE target2, OPERATOR OP, CODE *raw_data, BITS* compare_bitmap = NULL)
{
  for(int i = 0; i < bindex->length; i++) {
    bool hit = false;
    switch (OP)
    {
    case LT:
      if (raw_data[i] < target1) hit = true;
      break;
    case LE:
      if (raw_data[i] <= target1) hit = true;
      break;
    case GT:
      if (raw_data[i] > target1) hit = true;
      break;
    case GE:
      if (raw_data[i] >= target1) hit = true;
      break;
    case EQ:
      if (raw_data[i] == target1) hit = true;
      break;
    case BT:
      if (raw_data[i] > target1 && raw_data[i] < target2) hit = true;
      break;
    default:
      break;
    }
    if (hit) {
      // bitmap[i >> BITSSHIFT] |= (1U << (BITSWIDTH - 1 - i % BITSWIDTH));
      if (compare_bitmap != NULL) {
        int compare_bit = (compare_bitmap[i >> BITSSHIFT] & (1U << (BITSWIDTH - 1 - i % BITSWIDTH)));
        if(compare_bitmap == 0) {
          printf("[ERROR] check error in raw_data[%d]=", i);
          printf(" %u\n", raw_data[i]);
          break;
        }
      } else {
        refine(bitmap, i);
      }
    }
  }
}

void raw_scan_entry(std::vector<CODE>* target_l, std::vector<CODE>* target_r, std::string search_cmd, BinDex* bindex, BITS* bitmap, BITS* mergeBitmap, CODE* raw_data) {
  CODE target1, target2;
  
  for (int pi = 0; pi < target_l->size(); pi++) {
    target1 = (*target_l)[pi];
    if (target_r->size() != 0) {
      assert(search_cmd == "bt");
      target2 = (*target_r)[pi];
    }
    if (search_cmd == "bt" && target1 > target2) {
      std::swap(target1, target2);
    }

    if (search_cmd == "lt") {
      raw_scan(bindex, bitmap, target1, 0, LT, raw_data);
    } else if (search_cmd == "le") {
      raw_scan(bindex, bitmap, target1, 0, LE, raw_data);
    } else if (search_cmd == "gt") {
      raw_scan(bindex, bitmap, target1, 0, GT, raw_data);
    } else if (search_cmd == "ge") {
      raw_scan(bindex, bitmap, target1, 0, GE, raw_data);
    } else if (search_cmd == "eq") {
      raw_scan(bindex, bitmap, target1, 0, EQ, raw_data);
    } else if (search_cmd == "bt") {
      raw_scan(bindex, bitmap, target1, target2, BT, raw_data);
    }
  }

  /* for (int bindex_id = 1; bindex_id < bindex_num; bindex_id++) {
    for (int m = 0; m < bitmap_len; m++) {
      bitmap[0][m] = bitmap[0][m] & bitmap[m];
    }
  } */

  // CPU merge
  // int stride = 156250;
  // if (N / stride / CODEWIDTH > THREAD_NUM) {
  //   stride = N / THREAD_NUM / CODEWIDTH;
  //   printf("No enough threads, set stride to %d\n", stride);
  // }
  
  int max_idx = (N + CODEWIDTH - 1) / CODEWIDTH;
  int stride = (max_idx + THREAD_NUM - 1) / THREAD_NUM;

  if (mergeBitmap != bitmap) {
    std::thread threads[THREAD_NUM];
    int start_idx = 0;
    int end_idx = 0;
    size_t t_id = 0;
    while (end_idx < max_idx && t_id < THREAD_NUM) {
      end_idx = start_idx + stride;
      if (end_idx > max_idx) {
        end_idx = max_idx;
      }
      // printf("start idx: %d end idx: %d\n",start_idx, end_idx);
      // refine_result_bitmap(mergeBitmap, bitmap, start_idx, end_idx, t_id);
      threads[t_id] = std::thread(refine_result_bitmap, mergeBitmap, bitmap, start_idx, end_idx, t_id);
      start_idx += stride;
      t_id += 1;
    }
    for (int i = 0; i < THREAD_NUM; i++)
      threads[i].join();
  }
}

void compare_bitmap(BITS *bitmap_a, BITS *bitmap_b, int len, CODE **raw_data, int bindex_num)
{
  int total_hit = 0;
  int true_hit = 0;
  for (int i = 0; i < len; i++) {
    int data_a = (bitmap_a[i >> BITSSHIFT] & (1U << (BITSWIDTH - 1 - i % BITSWIDTH)));
    int data_b = (bitmap_b[i >> BITSSHIFT] & (1U << (BITSWIDTH - 1 - i % BITSWIDTH)));
    if (data_a) {
      total_hit += 1;
      if (data_b) true_hit += 1;
    }
    if (data_a != data_b) {
      printf("[ERROR] check error in raw_data[%d]=", i);
      printf(" %u / %u / %u \n", raw_data[0][i], raw_data[1][i], raw_data[2][i]);
      printf("the correct is %#x, but we have %#x\n", data_a, data_b);
      break;
    }
  }
  printf("[CHECK]hit %d/%d\n", true_hit, total_hit);
}

void start_timer(struct timeval* t) {
    gettimeofday(t, NULL);
}

void stop_timer(struct timeval* t, double* elapsed_time) {
    struct timeval end;
    gettimeofday(&end, NULL);
    *elapsed_time += (end.tv_sec - t->tv_sec) * 1000.0 + (end.tv_usec - t->tv_usec) / 1000.0;
}

void exp_opt(int argc, char *argv[]) {
  printf("N = %d\n", N);
  printf("K = %d\n", K);

  char opt;
  int selectivity;
  CODE target1, target2;
  char DATA_PATH[256] = "\0";
  char OPERATOR_TYPE[5];
  int insert_num = 0;
  int bindex_num = 1;
  bool USEKEYBOARDINPUT = false;

  // get command line options
  bool STRICT_SELECTIVITY = false;
  bool TEST_INSERTING = false;
  while ((opt = getopt(argc, argv, "khsil:r:o:f:n:p:b:")) != -1) {
    switch (opt) {
      case 'h':
        printf(
            "Usage: %s \n"
            "[-l <left target list>] [-r <right target list>]"
            "[-s use stric selectivity]"
            "[-i test inserting]"
            "[-p <scan-file>]"
            "[-f <input-file>] [-o <operator>] \n",
            argv[0]);
        exit(0);
      case 'l':
        target_numbers_l = get_target_numbers(optarg);
        break;
      case 'r':
        target_numbers_r = get_target_numbers(optarg);
        break;
      case 'o':
        strcpy(OPERATOR_TYPE, optarg);
        break;
      case 'f':
        strcpy(DATA_PATH, optarg);
        break;
      case 'n':
        // DATA_N
        insert_num = atoi(optarg);
        break;
      case 'b':
        bindex_num = atoi(optarg);
        break;
      case 'p':
        // prefetch_stride = str2uint32(optarg);
        strcpy(scan_file, optarg);
        break;
      case 's':
        STRICT_SELECTIVITY = true;
        break;
      case 'i':
        TEST_INSERTING = true;
        break;
      case 'k':
        USEKEYBOARDINPUT = true;
        break;
      default:
        printf("Error: unknown option %c\n", (char)opt);
        exit(-1);
    }
  }
  assert(target_numbers_r.size() == 0 || target_numbers_l.size() == target_numbers_r.size());
  assert(blockNumMax);
  assert(bindex_num >= 1);

  // initial data
  CODE *initial_data[MAX_BINDEX_NUM];


  if (!strlen(DATA_PATH)) {
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++)
    {
      initial_data[bindex_id] = (CODE *)malloc(N * sizeof(CODE));
      CODE *data = initial_data[bindex_id];
      printf("initing data by random\n");
      std::random_device rd;
      std::mt19937 mt(rd());
      std::uniform_int_distribution<CODE> dist(MINCODE, MAXCODE);
      CODE mask = ((uint64_t)1 << (sizeof(CODE) * 8)) - 1;
      for (int i = 0; i < N; i++)
      {
        data[i] = dist(mt) & mask;
        assert(data[i] <= mask);
      }
    }
  } else {
    FILE *fp;

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
        uint32_t min_val = UINT32_MAX, max_val = 0;
        for (int i = 0; i < N; i++) {
          data[i] = file_data[i];
          if (data[i] < min_val) min_val = data[i];
          if (data[i] > max_val) max_val = data[i];
        }
        free(file_data);
        printf("min_val: %u, max_val: %u\n", min_val, max_val);
      } else {
        printf("init_data_from_file: CODE_WIDTH != 8/16/32.\n");
        exit(-1);
      }
      printf("[CHECK] col %d  first num: %u  last num: %u\n", bindex_id, initial_data[bindex_id][0], initial_data[bindex_id][N - 1]);
    }

    fclose(fp);
  }
  
  BinDex *bindexs[MAX_BINDEX_NUM];
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    // Build the bindex structure
    printf("Build the bindex structure %d...\n", bindex_id);
    CODE *data = initial_data[bindex_id];
    bindexs[bindex_id] = (BinDex *)malloc(sizeof(BinDex));
    if (DEBUG_TIME_COUNT) timer.commonGetStartTime(0);
    PRINT_EXCECUTION_TIME("BinDex building", init_bindex(bindexs[bindex_id], data, N));
    if (DEBUG_TIME_COUNT) timer.commonGetEndTime(0);
    printf("\n");
  }

  // BinDex Scan
  printf("BinDex scan...\n");

  // result
  BITS *bitmap[MAX_BINDEX_NUM];
  int bitmap_len;
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    bitmap_len = bits_num_needed(bindexs[bindex_id]->length);
    bitmap[bindex_id] = (BITS *)aligned_alloc(SIMD_ALIGEN, bitmap_len * sizeof(BITS));
    memset_mt(bitmap[bindex_id], 0xFF, bitmap_len);
  }

  ifstream fin;
  fin.open(scan_file);
  if (!fin.is_open()) {
    cerr << "Fail to open scan_file!" << endl;
    exit(-1);
  }
  int toExit = 1;
  while(toExit != -1) {
    std::vector<CODE> target_l[MAX_BINDEX_NUM];
    std::vector<CODE> target_r[MAX_BINDEX_NUM]; 
    string search_cmd[MAX_BINDEX_NUM];

    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      cout << "input [operator] [target_l] [target_r] (" << bindex_id + 1 << "/" << bindex_num << ")" << endl;
      string input;

      if (USEKEYBOARDINPUT) {
        getline(cin, input);
      } else {
        getline(fin, input);
      }
      cout << input << endl;
      std::vector<std::string> cmds = stringSplit(input, ' ');
      if (cmds[0] == "exit") exit(0);
      search_cmd[bindex_id] = cmds[0];
      if (cmds.size() > 1) {
        target_l[bindex_id] = get_target_numbers(cmds[1]);
      }
      if (cmds.size() > 2) {
        target_r[bindex_id] = get_target_numbers(cmds[2]);
      }
    }
    timer.commonGetStartTime(11);
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      for (int pi = 0; pi < target_l[bindex_id].size(); pi++) {
        printf("RUNNING %d\n", pi);
        target1 = target_l[bindex_id][pi];
        if (target_r[bindex_id].size() != 0) {
          assert(search_cmd[bindex_id] == "bt");
          target2 = target_r[bindex_id][pi];
        }
        if (search_cmd[bindex_id] == "bt" && target1 > target2) {
          std::swap(target1, target2);
        }

        for (int i = 0; i < RUNS; i++) {
          if (search_cmd[bindex_id] == "lt") {
            PRINT_EXCECUTION_TIME("lt", bindex_scan_lt(bindexs[bindex_id], bitmap[bindex_id], target1));
            // check(bindexs[bindex_id], bitmap[bindex_id], target1, 0, LT, raw_datas[bindex_id]);
          } else if (search_cmd[bindex_id] == "le") {
            PRINT_EXCECUTION_TIME("le", bindex_scan_le(bindexs[bindex_id], bitmap[bindex_id], target1));
            // check(bindexs[bindex_id], bitmap[bindex_id], target1, 0, LE, raw_datas[bindex_id]);
          } else if (search_cmd[bindex_id] == "gt") {
            PRINT_EXCECUTION_TIME("gt", bindex_scan_gt(bindexs[bindex_id], bitmap[bindex_id], target1));
            // check(bindexs[bindex_id], bitmap[bindex_id], target1, 0, GT, raw_datas[bindex_id]);
          } else if (search_cmd[bindex_id] == "ge") {
            PRINT_EXCECUTION_TIME("ge", bindex_scan_ge(bindexs[bindex_id], bitmap[bindex_id], target1));
            // check(bindexs[bindex_id], bitmap[bindex_id], target1, 0, GE, raw_datas[bindex_id]);
          } else if (search_cmd[bindex_id] == "eq") {
            PRINT_EXCECUTION_TIME("eq", bindex_scan_eq(bindexs[bindex_id], bitmap[bindex_id], target1));
            // check(bindexs[bindex_id], bitmap[bindex_id], target1, 0, EQ, raw_datas[bindex_id]);
          } else if (search_cmd[bindex_id] == "bt") {
            PRINT_EXCECUTION_TIME("bt", bindex_scan_bt(bindexs[bindex_id], bitmap[bindex_id], target1, target2));
            // check(bindexs[bindex_id], bitmap[bindex_id], target1, target2, BT, raw_datas[bindex_id]);
          }

          printf("\n");
        }
      }
    }

    /* for (int bindex_id = 1; bindex_id < bindex_num; bindex_id++) {
      for (int m = 0; m < bitmap_len; m++) {
        bitmap[0][m] = bitmap[0][m] & bitmap[bindex_id][m];
      }
    } */
    
    
    // int stride = 156250;
    // if (N / stride / CODEWIDTH > THREAD_NUM) {
    //   stride = N / THREAD_NUM + 1;
    //   printf("No enough threads, set stride to %d\n",stride);
    // }

    int max_idx = (N + CODEWIDTH - 1) / CODEWIDTH;
    int stride = (max_idx + THREAD_NUM - 1) / THREAD_NUM;

    std::thread threads[THREAD_NUM];
    for (int bindex_id = 1; bindex_id < bindex_num; bindex_id++) {
      int start_idx = 0;
      int end_idx = 0;
      size_t t_id = 0;
      while(end_idx < max_idx && t_id < THREAD_NUM) {
        end_idx = start_idx + stride;
        if (end_idx > max_idx) {
          end_idx = max_idx;
        }
        threads[t_id] = std::thread(refine_result_bitmap, bitmap[0], bitmap[bindex_id], start_idx, end_idx, t_id);
        start_idx += stride;
        t_id += 1;
      }
      for (int i = 0; i < THREAD_NUM; i++) threads[i].join();
    }

    timer.commonGetEndTime(11);
    timer.showTime();
    timer.clear();

    // check jobs
    BITS *check_bitmap[MAX_BINDEX_NUM];
    int bitmap_len;
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      bitmap_len = bits_num_needed(bindexs[bindex_id]->length);
      check_bitmap[bindex_id] = (BITS *)aligned_alloc(SIMD_ALIGEN, bitmap_len * sizeof(BITS));
      memset_mt(check_bitmap[bindex_id], 0x0, bitmap_len);
    }

    // check final result 
    printf("[CHECK]check final result.\n");
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      raw_scan_entry(
          &(target_l[bindex_id]),
          &(target_r[bindex_id]),
          search_cmd[bindex_id],
          bindexs[bindex_id],
          check_bitmap[bindex_id],
          check_bitmap[0],
          initial_data[bindex_id]);
    }

    compare_bitmap(check_bitmap[0], bitmap[0], bindexs[0]->length, initial_data, bindex_num);
    printf("[CHECK]check final result done.\n\n");

    for (int i = 0; i < bitmap_len; i++) {
      if (bitmap[0][i] != 0) {
        printf("[CHECK] not all 0!\n");
        break;
      }
    }
  }

  // CAUTION! MAX_VAL * selc doesn't strictly match the selectivity, so we reset target_numbers_l with sorted data to
  // fix the selectivity
  /* if (STRICT_SELECTIVITY) {
    int n_selc = target_numbers_l.size();
    target_numbers_l.clear();
    for (int i = 0; i < n_selc - 1; i++) {
      target_numbers_l.push_back(data_sorted[(int)(DATA_N * i / (n_selc - 1))]);
    }
    target_numbers_l.push_back(data_sorted[(int)DATA_N - 1]);
  } */

  // clean jobs
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    free(initial_data[bindex_id]);
    free_bindex(bindexs[bindex_id], initial_data[bindex_id]);
    free(bitmap[bindex_id]);
  }
  // free(result1);
  // free(result2);
}

int main(int argc, char *argv[]) { exp_opt(argc, argv); }