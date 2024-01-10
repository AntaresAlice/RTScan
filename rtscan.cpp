#include "rt.h"

Timer timer;

std::mutex scan_refine_mutex;
int scan_refine_in_position;
CODE scan_selected_compares[MAX_BINDEX_NUM][2];
bool scan_skip_refine;
bool scan_skip_other_face[MAX_BINDEX_NUM];
bool scan_skip_this_face[MAX_BINDEX_NUM];
CODE scan_max_compares[MAX_BINDEX_NUM][2];
bool scan_inverse_this_face[MAX_BINDEX_NUM];

int density_width = 1200, density_height = 1200;
int default_ray_segment_num = 64;
int default_ray_length = 48000000;
int default_ray_mode = 1; // 0 for continuous ray, 1 for ray with space, 2 for ray as point

CODE min_val = UINT32_MAX;
CODE max_val = 0;

int NUM_QUERIES;
int RANGE_QUERY = 1;
int REDUCED_SCANNING = 0;
uint32_t data_range_list[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
int cube_width = 0;
bool READ_QUERIES_FROM_FILE = true;
int face_direction = 1; // 0: launch rays from wide face, 1: narrow face
bool with_refine = true;

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::string s;
    s.append(1, delim);
    std::regex reg(s);
    std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   std::sregex_token_iterator());
    return elems;
}

void start_timer(struct timeval* t) {
  gettimeofday(t, NULL);
}

void stop_timer(struct timeval* t, double* elapsed_time) {
  struct timeval end;
  gettimeofday(&end, NULL);
  *elapsed_time += (end.tv_sec - t->tv_sec) * 1000.0 + (end.tv_usec - t->tv_usec) / 1000.0;
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

void set_fv_val_less(BITS *bitmap, const CODE *val, CODE compare, POSTYPE n) {
  // Set values for filter vectors
  int i;
  for (i = 0; i + BITSWIDTH < (int)n; i += BITSWIDTH) {
    bitmap[i / BITSWIDTH] = gen_less_bits(val + i, compare, BITSWIDTH);
  }
  bitmap[i / BITSWIDTH] = gen_less_bits(val + i, compare, n - i);
}

void init_bindex_in_GPU(BinDex *bindex, CODE *data, POSTYPE n, int bindex_id, POSTYPE *pos, CODE *data_sorted) {
  bindex->length = n;
  POSTYPE avgAreaSize = n / K;
  cudaError_t cudaStatus;
  POSTYPE areaStartIdx[K];

#if ENCODE == 0
  data_sorted = (CODE *)malloc(n * sizeof(CODE));  // Sorted codes
#endif

  timeval start;
  double elapsed_time = 0;

#if ENCODE == 0
  start_timer(&start);
  pos = argsort(data, n);
  stop_timer(&start, &elapsed_time);
  std::cerr << "Sort[" << bindex_id << "]: " << elapsed_time << std::endl;
  std::cout << "Sort[" << bindex_id << "]: " << elapsed_time << std::endl;

  elapsed_time = 0.0;
  start_timer(&start);
  for (int i = 0; i < n; i++) {
    data_sorted[i] = data[pos[i]];
  }
  stop_timer(&start, &elapsed_time);
  std::cerr << "Assign[" << bindex_id << "]: " << elapsed_time << std::endl;
  std::cout << "Assign[" << bindex_id << "]: " << elapsed_time << std::endl;
#endif

  bindex->data_min = data_sorted[0];
  bindex->data_max = data_sorted[bindex->length - 1];
  
  printf("Bindex data min: %u  max: %u\n", bindex->data_min, bindex->data_max);

  bindex->areaStartValues[0] = data_sorted[0];
  areaStartIdx[0] = 0;

  elapsed_time = 0.0;
  start_timer(&start);
  for (int i = 1; i < K; i++) {
    bindex->areaStartValues[i] = data_sorted[i * avgAreaSize];
    int j = i * avgAreaSize;
    // if (bindex->areaStartValues[i] == bindex->areaStartValues[i - 1]) {
    if (ifEncodeEqual(bindex->areaStartValues[i], bindex->areaStartValues[i - 1], bindex_id)) {
      areaStartIdx[i] = j;
    } else {
      // To find the first element which is less than startValue
      // while (data_sorted[j] == bindex->areaStartValues[i]) {
      while (ifEncodeEqual(data_sorted[j], bindex->areaStartValues[i], bindex_id)) {
        j--;
      }
      areaStartIdx[i] = j + 1;
      bindex->areaStartValues[i] = data_sorted[j + 1];
    }
    // if(DEBUG_INFO) printf("area[%u] = %u\n", i, bindex->areaStartValues[i]);
  }
  stop_timer(&start, &elapsed_time);
  std::cerr << "Set areaStartValues[" << bindex_id << "]: " << elapsed_time << std::endl;
  std::cout << "Set areaStartValues[" << bindex_id << "]: " << elapsed_time << std::endl;
  
  std::thread threads[THREAD_NUM];

  elapsed_time = 0.0;
  start_timer(&start);
  // Build the filterVectors
  // Now we build them in CPU memory and then copy them to GPU memory
  for (int k = 0; k * THREAD_NUM < K - 1; k++) {
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      // Malloc 2 times of space, prepared for future appending
      bindex->filterVectors[k * THREAD_NUM + j] =
          (BITS *)aligned_alloc(SIMD_ALIGEN, bits_num_needed(n) * sizeof(BITS));
      threads[j] = std::thread(set_fv_val_less, bindex->filterVectors[k * THREAD_NUM + j], data,
                               bindex->areaStartValues[k * THREAD_NUM + j + 1], n);
    }
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      threads[j].join();
    }
  }
  stop_timer(&start, &elapsed_time);
  std::cerr << "Build FV[" << bindex_id << "]: " << elapsed_time << std::endl;
  std::cout << "Build FV[" << bindex_id << "]: " << elapsed_time << std::endl;

  for (int i = 0; i < K - 1; i++) {
    cudaStatus = cudaMalloc((void**)&(bindex->filterVectorsInGPU[i]), bits_num_needed(n) * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(-1);
    }

    timer.commonGetStartTime(24);
    cudaStatus = cudaMemcpy(bindex->filterVectorsInGPU[i], bindex->filterVectors[i], bits_num_needed(n) * sizeof(BITS), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        exit(-1);
    }
    timer.commonGetEndTime(24);

    free(bindex->filterVectors[i]);
  }

  free(pos);
  free(data_sorted);
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

void copy_filter_vector_in_GPU(BinDex *bindex, BITS *dev_bitmap, int k, bool negation = false) {
  int bitmap_len = bits_num_needed(bindex->length);

  if (k < 0) {
    cudaMemset(dev_bitmap, 0, bitmap_len * sizeof(BITS));
    return;
  }

  if (k >= (K - 1)) {
    cudaMemset(dev_bitmap, 0xFF, bitmap_len * sizeof(BITS));
    return;
  }

  if (!negation)
    GPUbitCopyWithCuda(dev_bitmap, bindex->filterVectorsInGPU[k], bitmap_len);
  else
    GPUbitCopyNegationWithCuda(dev_bitmap, bindex->filterVectorsInGPU[k], bitmap_len);
}

void copy_filter_vector_bt_in_GPU(BinDex *bindex, BITS *result, int kl, int kr) {
  int bitmap_len = bits_num_needed(bindex->length);

  if (kr < 0) {
    // assert(0);
    // printf("1\n");
    // memset_mt(result, 0, bitmap_len);
    cudaMemset(result, 0, bitmap_len * sizeof(BITS));
    return;
  } else if (kr >= (K - 1)) {
    // assert(0);
    // printf("2\n");
    // copy_filter_vector_not(bindex, result, kl);
    copy_filter_vector_in_GPU(bindex, result, kl, true);
    return;
  }
  if (kl < 0) {
    // assert(0);
    // printf("3\n");
    // copy_filter_vector(bindex, result, kr);
    copy_filter_vector_in_GPU(bindex, result, kr);
    return;
  } else if (kl >= (K - 1)) {
    // assert(0);
    // printf("4\n");
    // memset_mt(result, 0, bitmap_len);  // Slower(?) than a loop
    cudaMemset(result, 0, bitmap_len * sizeof(BITS));
    return;
  }

  GPUbitCopySIMDWithCuda(result, 
                         bindex->filterVectorsInGPU[kl], 
                         bindex->filterVectorsInGPU[kr],
                         bitmap_len);
}

inline void refine(BITS *bitmap, POSTYPE pos) { bitmap[pos >> BITSSHIFT] ^= (1U << (BITSWIDTH - 1 - pos % BITSWIDTH)); }

void refine_result_bitmap(BITS *bitmap_a, BITS *bitmap_b, int start_idx, int end_idx, int t_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  int i;
  for (i = start_idx; i < end_idx; i++) {
    __sync_fetch_and_and(&bitmap_a[i], bitmap_b[i]);
  }
}

int find_appropriate_fv(BinDex *bindex, CODE compare) {
  if (compare < bindex->areaStartValues[0]) return -1;
  for (int i = 0; i < K; i++) {
    CODE area_sv = bindex->areaStartValues[i];
    if (area_sv == compare) return i;
    if (area_sv > compare) return i - 1;
  }
  // check if actually out of boundary here.
  // if so, return K
  // need extra check before use the return value avoid of error.
  if (compare <= bindex->data_max)
    return K - 1;
  else
    return K;
}

void bindex_scan_lt_in_GPU(BinDex *bindex, BITS *dev_bitmap, CODE compare, int bindex_id) {
  int bitmap_len = bits_num_needed(bindex->length);
  int area_idx = find_appropriate_fv(bindex, compare);

  // set common compare here for use in refine
  scan_max_compares[bindex_id][0] = bindex->data_min;
  scan_max_compares[bindex_id][1] = compare;

  if (area_idx < 0) {
    scan_refine_mutex.lock();
    scan_skip_refine = true;
    scan_refine_mutex.unlock();
    return;
  }

  if (area_idx == 0) {
    // 'compare' less than all raw_data, return all zero result
    scan_refine_mutex.lock();
    if(!scan_skip_refine) {
      scan_skip_other_face[bindex_id] = true;
      scan_selected_compares[bindex_id][0] = bindex->data_min;
      scan_selected_compares[bindex_id][1] = compare;
      // printf("comapre[%d]: %u %u\n", bindex_id, scan_selected_compares[bindex_id][0], scan_selected_compares[bindex_id][1]);
      scan_refine_in_position += 1;
    }
    scan_refine_mutex.unlock();
    return;
  }

  if (area_idx > K - 1) {
    // set skip this bindex so rt will skip the scan for this face
    scan_refine_mutex.lock();
    if(!scan_skip_refine) {
      scan_skip_this_face[bindex_id] = true;
      scan_selected_compares[bindex_id][0] = bindex->data_min;
      scan_selected_compares[bindex_id][1] = compare;
      scan_refine_in_position += 1;
    }
    scan_refine_mutex.unlock();
    return;
  }

  // choose use inverse or normal
  bool inverse = false;
  if (area_idx < K - 1) {
    if (compare - bindex->areaStartValues[area_idx] > bindex->areaStartValues[area_idx + 1] - compare) {
      inverse = true;
      scan_inverse_this_face[bindex_id] = true;
    }
  }
  printf("bindex->areaStartValues[%d]=%u\n", area_idx, bindex->areaStartValues[area_idx]);

  // set refine compares here
  scan_refine_mutex.lock();
  if (inverse) {
    scan_selected_compares[bindex_id][0] = compare;
    scan_selected_compares[bindex_id][1] = bindex->areaStartValues[area_idx + 1];
  } else {
    scan_selected_compares[bindex_id][0] = bindex->areaStartValues[area_idx];
    scan_selected_compares[bindex_id][1] = compare;
  }

  if(DEBUG_INFO) printf("area [%d]\n", area_idx);
  if(DEBUG_INFO) printf("comapre[%d]: %u %u\n", bindex_id, scan_selected_compares[bindex_id][0], scan_selected_compares[bindex_id][1]);
  scan_refine_in_position += 1;
  scan_refine_mutex.unlock();

  // we use the one small than compare here, so rt must return result to append (maybe with and)
  if(!scan_skip_refine) {
    if (inverse) {
      PRINT_EXCECUTION_TIME("copy",
                            copy_filter_vector_in_GPU(bindex, dev_bitmap, area_idx))
    }
    else {
      PRINT_EXCECUTION_TIME("copy",
                            copy_filter_vector_in_GPU(bindex, dev_bitmap, area_idx - 1))
    }
  }
}

void bindex_scan_gt_in_GPU(BinDex *bindex, BITS *dev_bitmap, CODE compare, int bindex_id) {
  compare = compare + 1;

  int bitmap_len = bits_num_needed(bindex->length);
  int area_idx = find_appropriate_fv(bindex, compare);

  // set common compare here for use in refine
  scan_max_compares[bindex_id][0] = compare;
  scan_max_compares[bindex_id][1] = bindex->data_max;
  
  // handle some boundary problems: <0, ==0, K-1, >K-1 (just like lt)
  // <0: set all bits to 1. skip this face: CS[this bindex] x CM[other bindexs]
  // ==0: may not cause problem here?
  // K-1: set all bits to 0. just scan one face: CS[this bindex] x CM[other bindexs].
  // K: set all bits to 0. should not call rt. interrupt other procedures now. should cancel merge as well?
  
  if (area_idx > K - 1) {
    // set skip to true so refine thread will be informed to skip
    scan_refine_mutex.lock();
    scan_skip_refine = true;
    scan_refine_mutex.unlock();
    return;
  }

  if (area_idx == K - 1) {
    // 'compare' less than all raw_data, return all zero result
    scan_refine_mutex.lock();
    if(!scan_skip_refine) {
      scan_skip_other_face[bindex_id] = true;
      scan_selected_compares[bindex_id][0] = compare;
      scan_selected_compares[bindex_id][1] = bindex->data_max;
      scan_refine_in_position += 1;
    }
    scan_refine_mutex.unlock();
    return;
  }
  
  if (area_idx < 0) {
    // 'compare' less than all raw_data, return all 1 result
    scan_refine_mutex.lock();
    if(!scan_skip_refine) {
      scan_skip_this_face[bindex_id] = true;
      scan_selected_compares[bindex_id][0] = compare;
      scan_selected_compares[bindex_id][1] = bindex->data_max;
      scan_refine_in_position += 1;
    }
    scan_refine_mutex.unlock();
    return;
  }

  // set refine compares here
  scan_refine_mutex.lock();
  scan_selected_compares[bindex_id][0] = compare;
  if (area_idx + 1 < K)
    scan_selected_compares[bindex_id][1] = bindex->areaStartValues[area_idx + 1];
  else 
    scan_selected_compares[bindex_id][1] = bindex->data_max;
  scan_refine_in_position += 1;
  scan_refine_mutex.unlock();

  // we use the one larger than compare here, so rt must return result to append (maybe with and)    retrun range -> | RT Scan | Bindex filter vector |
  PRINT_EXCECUTION_TIME("copy",
                        copy_filter_vector_in_GPU(bindex, dev_bitmap, area_idx, true))

  
}

void bindex_scan_bt_in_GPU(BinDex *bindex, BITS *result, CODE compare1, CODE compare2, int bindex_id) {
  assert(compare2 > compare1);
  compare1 = compare1 + 1;

  int bitmap_len = bits_num_needed(bindex->length);

  // x > compare1
  int area_idx_l = find_appropriate_fv(bindex, compare1);;
  if (area_idx_l < 0) {
    bindex_scan_lt_in_GPU(bindex, result, compare2, bindex_id);
    return;
  }
  
  bool is_upper_fv_l = false;
  if (area_idx_l < K - 1) {
    if (compare1 - bindex->areaStartValues[area_idx_l] < bindex->areaStartValues[area_idx_l + 1] - compare1) {
      is_upper_fv_l = true;
      // scan_inverse_this_face[bindex_id] = true;
      scan_selected_compares[bindex_id][0] = bindex->areaStartValues[area_idx_l];
      scan_selected_compares[bindex_id][1] = compare1;
    } else {
      scan_selected_compares[bindex_id][0] = compare1;
      scan_selected_compares[bindex_id][1] = bindex->areaStartValues[area_idx_l + 1];
    }
  }

  // x < compare2
  int area_idx_r = find_appropriate_fv(bindex, compare2);
  bool is_upper_fv_r = false;
  if (area_idx_r < K - 1) {
    if (compare2 - bindex->areaStartValues[area_idx_r] < bindex->areaStartValues[area_idx_r + 1] - compare2) {
      is_upper_fv_r = true;
      // scan_inverse_this_face[bindex_id + 3] = true;
      scan_selected_compares[bindex_id + 3][0] = bindex->areaStartValues[area_idx_r];
      scan_selected_compares[bindex_id + 3][1] = compare2;
    } else {
      scan_selected_compares[bindex_id + 3][0] = compare2;
      scan_selected_compares[bindex_id + 3][1] = bindex->areaStartValues[area_idx_r + 1];
    }
  }


  PRINT_EXCECUTION_TIME("copy",
                        copy_filter_vector_bt_in_GPU(bindex,
                                              result,
                                              is_upper_fv_l ? (area_idx_l - 1) : (area_idx_l),
                                              is_upper_fv_r ? (area_idx_r - 1) : (area_idx_r))
                        )
}

void bindex_scan_eq_in_GPU(BinDex *bindex, BITS *result, CODE compare, int bindex_id) {

  // just set MC = SC = [compare - 1, compare + 1] and skip_face = true
  scan_refine_mutex.lock();
  scan_skip_this_face[bindex_id] = true;
  scan_max_compares[bindex_id][0] = compare - 1;
  scan_max_compares[bindex_id][1] = compare + 1;
  scan_selected_compares[bindex_id][0] = compare - 1;
  scan_selected_compares[bindex_id][1] = compare + 1;
  scan_refine_in_position += 1;
  scan_refine_mutex.unlock();
}

void free_bindex(BinDex *bindex) {
  free(bindex);
}

std::vector<CODE> get_target_numbers(const char *s) {
  std::string input(s);
  std::stringstream ss(input);
  std::string value;
  std::vector<CODE> result;
  while (std::getline(ss, value, ',')) {
    result.push_back((CODE)stod(value));
  }
  return result;
}

std::vector<CODE> get_target_numbers(string s) {
  std::stringstream ss(s);
  std::string value;
  std::vector<CODE> result;
  while (std::getline(ss, value, ',')) {
    result.push_back((CODE)stod(value));
  }
  return result;
}

// remember to free data ptr after using
void getDataFromFile(char *DATA_PATH, CODE **initial_data, int bindex_num) {
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
    for (int i = 0; i < N; i++)
      data[i] = data[i] & data_range_list[bindex_id];
    printf("[CHECK] col %d  first num: %u  last num: %u\n", bindex_id, initial_data[bindex_id][0], initial_data[bindex_id][N - 1]);
  }
}

void compare_bitmap(BITS *bitmap_a, BITS *bitmap_b, int len, CODE **raw_data, int bindex_num) {
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
      printf("the correct is %u, but we have %u\n", data_a, data_b);
      for (int j = 0; j < bindex_num; j++) {
        printf("SC[%d] = [%u,%u], MC[%d] = [%u,%u]\n",j,scan_selected_compares[j][0],scan_selected_compares[j][1],
        j,scan_max_compares[j][0],scan_max_compares[j][1]);
      }
      break;
    }
  }
  printf("[CHECK]hit %d/%d\n", true_hit, total_hit);
  return;
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
      threads[t_id] = std::thread(
        refine_result_bitmap, 
        mergeBitmap, bitmap, 
        start_idx, end_idx, t_id
      );
      start_idx += stride;
      t_id += 1;
    }
    for (int i = 0; i < THREAD_NUM; i++)
      threads[i].join();
  }
}

void scan_multithread_withGPU(std::vector<CODE> *target_l, std::vector<CODE> *target_r,  CODE target1, CODE target2, std::string search_cmd, BinDex *bindex, BITS *bitmap, int bindex_id)
{
  for (int pi = 0; pi < target_l->size(); pi++) {
    printf("RUNNING %d\n", pi);
    target1 = (*target_l)[pi];
    if (target_r->size() != 0) {
      assert(search_cmd == "bt");
      target2 = (*target_r)[pi];
    }
    if (search_cmd == "bt" && target1 > target2) {
      std::swap(target1, target2);
    }

    if (search_cmd == "lt") {
      PRINT_EXCECUTION_TIME("lt", bindex_scan_lt_in_GPU(bindex, bitmap, target1, bindex_id));
    } else if (search_cmd == "le") {
      PRINT_EXCECUTION_TIME("le", bindex_scan_lt_in_GPU(bindex, bitmap, target1 + 1, bindex_id));
    } else if (search_cmd == "gt") {
      PRINT_EXCECUTION_TIME("gt", bindex_scan_gt_in_GPU(bindex, bitmap, target1, bindex_id));
    } else if (search_cmd == "ge") {
      PRINT_EXCECUTION_TIME("ge", bindex_scan_gt_in_GPU(bindex, bitmap, target1 - 1, bindex_id));
    } else if (search_cmd == "eq") {
      PRINT_EXCECUTION_TIME("eq", bindex_scan_eq_in_GPU(bindex, bitmap, target1, bindex_id));
    } else if (search_cmd == "bt") {
      PRINT_EXCECUTION_TIME("bt", bindex_scan_bt_in_GPU(bindex, bitmap, target1, target2, bindex_id));
    }

    printf("\n");
  }
}

int calculate_ray_segment_num(int direction, double *predicate, BinDex **bindexs, int best_ray_num)
{
  int width = density_width;
  int height = density_height;
  int launch_width;
  int launch_height;
  if (direction == 2) {
      launch_width  = static_cast<int>((predicate[1] - predicate[0]) * width / (bindexs[0]->data_max - bindexs[0]->data_min)) + 1;
      launch_height = static_cast<int>((predicate[3] - predicate[2]) * height / (bindexs[1]->data_max - bindexs[1]->data_min)) + 1;
  } else if (direction == 1) {
      launch_width  = static_cast<int>((predicate[1] - predicate[0]) * width / (bindexs[0]->data_max - bindexs[0]->data_min)) + 1;
      launch_height = static_cast<int>((predicate[5] - predicate[4]) * height / (bindexs[2]->data_max - bindexs[2]->data_min)) + 1;
  } else {
      launch_width  = static_cast<int>((predicate[3] - predicate[2]) * width / (bindexs[1]->data_max - bindexs[1]->data_min)) + 1;
      launch_height = static_cast<int>((predicate[5] - predicate[4]) * height / (bindexs[2]->data_max - bindexs[2]->data_min)) + 1;
  }
  printf("[LOG] launch width: %d launch height: %d\n",launch_width, launch_height);
  int ray_segment_num = best_ray_num / launch_width / launch_height;
  printf("[LOG] ray_segment_num: %d\n",ray_segment_num);
  if (ray_segment_num <= 0) {
    return 1;
  }
  else {
    return ray_segment_num;
  }
}

void special_eq_scan(CODE *target_l, CODE *target_r, BinDex **bindexs, BITS *dev_bitmap, const int bindex_num, string *search_cmd)
{
  if(DEBUG_TIME_COUNT) timer.commonGetStartTime(13);
  if (DEBUG_INFO) {
    printf("[INFO] use special eq scan\n");
  }
  double **compares = (double **)malloc(bindex_num * sizeof(double *));
  double *dev_predicate = (double *)malloc(bindex_num * 2 * sizeof(double));
  for (int i = 0; i < bindex_num; i++) {
    compares[i] = &(dev_predicate[i * 2]);
  }

  // prepare MC and SC
  int direction = 0;
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if (search_cmd[bindex_id] == "eq") {
      compares[bindex_id][0] = double(target_l[bindex_id]) - 1.0;
      if (compares[bindex_id][0] < 0) compares[bindex_id][0] = 0;
      compares[bindex_id][1] = double(target_l[bindex_id]) + 2.0;

      direction = bindex_id;
    } else if (search_cmd[bindex_id] == "lt") {
      compares[bindex_id][0] = bindexs[bindex_id]->data_min - 1.0;
      compares[bindex_id][1] = double(target_l[bindex_id]);
      if (compares[bindex_id][0] > compares[bindex_id][1]) {
        swap(compares[bindex_id][0],compares[bindex_id][1]);
      }
    } else if (search_cmd[bindex_id] == "gt") {
      compares[bindex_id][0] = double(target_l[bindex_id]);
      compares[bindex_id][1] = bindexs[bindex_id]->data_max + 1.0;
    } else {
      printf("[ERROR] not support yet!\n");
      exit(-1);
    }
  }
  
  if(DEBUG_INFO) {
    for (int i = 0; i < 6; i++) {
        printf("%f ", dev_predicate[i]);
    }
    printf("\n");
    printf("direction = %d\n", direction);
    printf("ray segment num = %d\n", default_ray_segment_num);
    printf("[INFO] compares prepared.\n");
  }
  if (with_refine) {
    refineWithOptix(dev_bitmap, dev_predicate, bindex_num, default_ray_length, default_ray_segment_num, false, direction, default_ray_mode);
  }
  if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
}

int get_refine_space_side(double **compares, int bindex_num, int face_direction) {
  double dis = compares[0][1] - compares[0][0];
  int d = 0;
  for (int bindex_id = 1; bindex_id < bindex_num; bindex_id++) {
    if (face_direction == 0) { // wide face, short side
      if (compares[bindex_id][1] - compares[bindex_id][0] < dis) {
        dis = compares[bindex_id][1] - compares[bindex_id][0];
        d = bindex_id;
      }
    } else { // narrow face, long side
      if (compares[bindex_id][1] - compares[bindex_id][0] > dis) {
        dis = compares[bindex_id][1] - compares[bindex_id][0];
        d = bindex_id;
      }
    }
  }
  return d;
}

void refine_with_GPU(BinDex **bindexs, BITS *dev_bitmap, const int bindex_num) {
  timer.commonGetStartTime(23);
  bool adjust_ray_num = false; // switch for fixed ray number: `true` for `fixed`
  int default_ray_total_num = 20000;
  int ray_length = default_ray_length; // set to -1 so rt will use default segmentnum set

  if(DEBUG_TIME_COUNT) timer.commonGetStartTime(18);
  double **compares = (double **)malloc(bindex_num * sizeof(double *));
  double *dev_predicate = (double *)malloc(bindex_num * 2 * sizeof(double));
  for (int i = 0; i < bindex_num; i++) {
    compares[i] = &(dev_predicate[i * 2]);
  }
  if(DEBUG_TIME_COUNT) timer.commonGetEndTime(18);
  
  if(DEBUG_TIME_COUNT) timer.commonGetStartTime(13);
  // if there is a compare totally out of boundary, refine procedure can be skipped
  if (scan_skip_refine) {
    if(DEBUG_INFO) printf("[INFO] Search out of boundary, skip all refine.\n");
    if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
    timer.commonGetEndTime(23);
    return;
  }

  // check if we can scan only one face
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if(scan_skip_other_face[bindex_id]) {
      if(DEBUG_INFO) printf("[INFO] %d face scan, other skipped.\n",bindex_id);
      // no matter use sc or mc
      compares[bindex_id][0] = scan_selected_compares[bindex_id][0];
      compares[bindex_id][1] = scan_selected_compares[bindex_id][1];

      for (int other_bindex_id = 0; other_bindex_id < bindex_num; other_bindex_id++) {
        if (other_bindex_id == bindex_id) continue;
        compares[other_bindex_id][0] = scan_max_compares[other_bindex_id][0];
        compares[other_bindex_id][1] = scan_max_compares[other_bindex_id][1];
      }

      for (int i = 0; i < bindex_num; i++) {
        if (compares[i][0] == compares[i][1]) {
          if(DEBUG_INFO) printf("[INFO] %d face scan skipped for the same compares[0] and compares[1].\n",bindex_id);
          if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
          timer.commonGetEndTime(23);
          return;
        }
      }

      int direction = get_refine_space_side(compares, bindex_num, face_direction);

      // calculate ray_segment_num
      if (adjust_ray_num) default_ray_segment_num = calculate_ray_segment_num(direction, dev_predicate, bindexs, default_ray_total_num);
      
      // Solve bound problem
      for (int i = 0; i < bindex_num; i++) {
        compares[i][0] -= 1;
      }
      
      // add refine here
      // send compares, dev_bitmap, the result is in dev_bitmap
      if(DEBUG_INFO) {
        printf("[Prepared predicate]");
        for (int i = 0; i < 6; i++) {
            printf("%f ", dev_predicate[i]);
        }
        printf("\n");
        printf("direction = %d\n", direction);
        printf("[INFO] compares prepared.\n");
      }
      timer.commonGetEndTime(23);
      refineWithOptix(dev_bitmap, dev_predicate, bindex_num, ray_length, default_ray_segment_num, false, direction, default_ray_mode);
      if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
      return;
    }
  }

  double selectivity = 0.0;
  // rt scan every face
  /// split inversed face and non-inversed face first
  std::vector<int> inversed_face;
  std::vector<int> normal_face;
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if (scan_inverse_this_face[bindex_id]) {
      inversed_face.push_back(bindex_id);
    }
    else {
      normal_face.push_back(bindex_id);
    }
  }
  /// start refine
  int max_MS_face_count = 0;
  for (int i = 0; i < bindex_num; i++) {
    double face_selectivity = 1.0;
    bool inverse = false;
    int bindex_id;
    if (normal_face.size() != 0) {
      bindex_id = normal_face[0];
      normal_face.erase(normal_face.begin());
    }
    else if (inversed_face.size() != 0) {
      bindex_id = inversed_face[0];
      inversed_face.erase(inversed_face.begin());
      inverse = true;
    }

    int current_MS_face_count = 0;
    if(scan_skip_this_face[bindex_id]) {
      if(DEBUG_INFO) printf("[INFO] %d face scan skipped.\n",bindex_id);
      continue;
    }
    // select SC face
    compares[bindex_id][0] = scan_selected_compares[bindex_id][0];
    compares[bindex_id][1] = scan_selected_compares[bindex_id][1];

    // revise S and C here to avoid a < x < b scan
    // compares[bindex_id][0] -= 1.0;
    for (int other_bindex_id = 0; other_bindex_id < bindex_num; other_bindex_id++) {
      if (other_bindex_id == bindex_id) continue;
      if (current_MS_face_count < max_MS_face_count) {
        CODE S;
        if (inverse) 
          S = scan_selected_compares[other_bindex_id][1];
        else
          S = scan_selected_compares[other_bindex_id][0];
        if (scan_max_compares[other_bindex_id][0] < S) {
          compares[other_bindex_id][0] = scan_max_compares[other_bindex_id][0];
          compares[other_bindex_id][1] = S;
        }
        else {
          compares[other_bindex_id][0] = S;
          compares[other_bindex_id][1] = scan_max_compares[other_bindex_id][0];
        }
        current_MS_face_count += 1;
      } else {
        compares[other_bindex_id][0] = scan_max_compares[other_bindex_id][0];
        compares[other_bindex_id][1] = scan_max_compares[other_bindex_id][1];
      }
    }

    bool mid_skip_flag = false;
    for (int j = 0; j < bindex_num; j++) {
      if (compares[j][0] == compares[j][1]) {
        if(DEBUG_INFO) printf("[INFO] %d face scan skipped for the same compares[0] and compares[1].\n",bindex_id);
        mid_skip_flag = true;
        break;
      }
    }
    if (mid_skip_flag) continue;

    if(DEBUG_INFO) {
      for (int i = 0; i < bindex_num; i++) {
        face_selectivity *= double(compares[i][1] - compares[i][0]) / double(bindexs[i]->data_max - bindexs[i]->data_min);
      }
      selectivity += face_selectivity;
    }

    int direction = get_refine_space_side(compares, bindex_num, face_direction);

    // calculate ray_segment_num
    if (adjust_ray_num) default_ray_segment_num = calculate_ray_segment_num(direction, dev_predicate, bindexs, default_ray_total_num);

    // Solve bound problem
    for (int i = 0; i < bindex_num; i++) {
      compares[i][0] -= 1;
    }

    // add refine here
    // send compares, dev_bitmap, the result is in dev_bitmap
    if(DEBUG_INFO) {
      printf("[Prepared predicate] MS = %d, predicate: ", max_MS_face_count);
      for (int i = 0; i < 6; i++) {
          printf("%f ", dev_predicate[i]);
      }
      printf("\n");
      printf("direction = %d\n", direction);
      printf("[INFO] compares prepared.\n");
    }
    timer.commonGetEndTime(23);
    refineWithOptix(dev_bitmap, dev_predicate, bindex_num, ray_length, default_ray_segment_num, inverse, direction, default_ray_mode);
    timer.commonGetStartTime(23);
    max_MS_face_count += 1;
  }
  timer.commonGetEndTime(23);
  if(DEBUG_INFO) printf("total selectivity: %f\n", selectivity);
  if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
}

void merge_with_GPU(BITS *merge_bitmap, BITS **dev_bitmaps, const int bindex_num, const int bindex_len) {
  timer.commonGetStartTime(15);

  int bitmap_len = bits_num_needed(bindex_len);
  if (scan_skip_refine) {
    if(DEBUG_INFO) printf("[INFO] Search out of boundary, skip all merge.\n");
    cudaMemset(merge_bitmap, 0, bitmap_len * sizeof(BITS));
    timer.commonGetEndTime(15);
    return;
  }

  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if(scan_skip_other_face[bindex_id]) {
      if(DEBUG_INFO) printf("[INFO] %d face merge, other skipped.\n",bindex_id);
      cudaMemset(merge_bitmap, 0, bitmap_len * sizeof(BITS));
      timer.commonGetEndTime(15);
      return;
    }
  }

  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if (merge_bitmap == dev_bitmaps[bindex_id]) {
      if (scan_skip_this_face[bindex_id]) {
        if(DEBUG_INFO) printf("[INFO] merge face %d skipped, set it to 0xFF\n", bindex_id);
        cudaMemset(merge_bitmap, 0xFF, bitmap_len * sizeof(BITS));
      }
      break;
    }
  }

  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if (!scan_skip_this_face[bindex_id]) {
      if (merge_bitmap != dev_bitmaps[bindex_id]) {
        GPUbitAndWithCuda(merge_bitmap, dev_bitmaps[bindex_id], bindex_len);
      }
    }
    else {
      if(DEBUG_INFO) printf("[INFO] %d face merge skipped.\n",bindex_id);
    }
  }

  timer.commonGetEndTime(15);
  return;
}

// Only support `lt` now.
void generate_range_queries(vector<CODE> &target_lower, 
                            vector<CODE> &target_upper,
                            vector<string> &search_cmd,
                            int column_num,
                            CODE *range) {
#ifndef RANGE_SELECTIVITY_11

#if DISTRITION == 0                              
  CODE span = UINT32_MAX / (NUM_QUERIES + 1);
#else
  CODE span = (max_val - min_val + NUM_QUERIES - 1) / NUM_QUERIES;
#endif

  target_lower.resize(NUM_QUERIES * column_num);
  search_cmd.resize(NUM_QUERIES * column_num);
  for (int i = 0; i < NUM_QUERIES; i++) {
    for (int column_id = 0; column_id < column_num; column_id++) {
      target_lower[i * column_num + column_id] = (i + 1) * span;
      search_cmd[i * column_num + column_id] = "lt";
    }
  }

#else
  double span[3] = {1.0 * data_range_list[0] / (NUM_QUERIES - 1), 
                    1.0 * data_range_list[1] / (NUM_QUERIES - 1), 
                    1.0 * data_range_list[2] / (NUM_QUERIES - 1)};
  target_lower.resize(NUM_QUERIES * column_num);
  search_cmd.resize(NUM_QUERIES * column_num);
  for (int i = 0; i < NUM_QUERIES; i++) {
    for (int column_id = 0; column_id < column_num; column_id++) {
      target_lower[i * column_num + column_id] = CODE(i * span[column_id]);
      if (i == NUM_QUERIES - 1) {
        if (range[2 * column_id + 1] != UINT32_MAX) {
          target_lower[i * column_num + column_id] = range[2 * column_id + 1] + 1;
        } else {
          target_lower[i * column_num + column_id] = UINT32_MAX;
        }
      }
      search_cmd[i * column_num + column_id] = "lt";
    }
  }
#endif

  for (int i = 0; i < NUM_QUERIES; i++) {
    for (int column_id = 0; column_id < column_num; column_id++) {
      printf("%s %u\n", search_cmd[i * column_num + column_id].c_str(), target_lower[i * column_num + column_id]);
    }
  }
}

void generate_point_queries(vector<CODE> &target_lower, 
                            vector<string> &search_cmd,
                            int column_num) {
#if DISTRITION == 0                              
  CODE span = UINT32_MAX / (NUM_QUERIES + 1);
#else
  CODE span = (max_val - min_val + NUM_QUERIES - 1) / NUM_QUERIES;
#endif

  target_lower.resize(NUM_QUERIES * column_num);
  search_cmd.resize(NUM_QUERIES * column_num);
  for (int i = 0; i < NUM_QUERIES; i++) {
    for (int column_id = 0; column_id < column_num; column_id++) {
      target_lower[i * column_num + column_id] = (i + 1) * span;
      search_cmd[i * column_num + column_id] = "eq";
    }
  }
}

size_t memory_used_by_process() {
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];
  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      int len = strlen(line);
      const char* p = line;
      for (; std::isdigit(*p) == false; ++p) {}
      line[len - 3] = 0;
      result = atoi(p);
      break;
    }
  }
  fclose(file);
  return result;  // KB
}

int main(int argc, char *argv[]) {
  printf("N = %d\n", N);
  printf("DISTRIBUTION: %d\n", DISTRIBUTION);
  printf("K: %d\n", K);
  
  char opt;
  int selectivity;
  CODE target1, target2;
  char DATA_PATH[256] = "\0";
  char SCAN_FILE[256] = "\0";
  char OPERATOR_TYPE[5];
  int insert_num = 0;
  int bindex_num = 3;

  while ((opt = getopt(argc, argv, "ha:b:c:d:e:f:g:w:m:o:p:q:s:u:v:y:z:")) != -1) {
    switch (opt) {
      case 'h':
        printf(
            "Usage: %s \n"
            "[-l <left target list>] [-r <right target list>]\n"
            "[-a <ray-length>]\n" // -1: flexible length controlled by ray_segment_num, -2: fixed length controlled by ray_segment_num
            "[-b <column-num>]\n"
            "[-c <reduced-scanning>]\n"
            "[-d <data-range>]\n"
            "[-e <ray-mode>]\n"
            "[-f <input-file>]\n"
            "[-g <range-query>]\n"
            "[-i test inserting]\n"
            "[-w <ray-range-width>] [-m <ray-range-height>]\n"
            "[-o <operator>]\n"
            "[-p <scan-predicate-file>]\n"
            "[-q <query-num>]\n"
            "[-s <ray-segment-num>]\n"
            "[-u <with-refine>]\n"
            "[-v <cube-width>]\n"
            "[-y <switch-for-reading-queries-from-file>]"
            "[-z <face-direction-0=wide-1=narrow>]\n",
            argv[0]);
        exit(0);
      case 'o':
        strcpy(OPERATOR_TYPE, optarg);
        break;
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
        default_ray_length = atoi(optarg);
        break;
      case 'e':
        default_ray_mode = atoi(optarg);
        break;
      case 'q':
        NUM_QUERIES = atoi(optarg);
        break;
      case 'g':
        RANGE_QUERY = atoi(optarg);
        break;
      case 'c':
        REDUCED_SCANNING = atoi(optarg);
        break;
      case 'd':
        sscanf(optarg, "%u,%u,%u", &data_range_list[0], &data_range_list[1], &data_range_list[2]);
        break;
      case 'u':
        with_refine = atoi(optarg);
        break;
      case 'v':
        cube_width = atoi(optarg);
        break;
      case 'y':
        READ_QUERIES_FROM_FILE = atoi(optarg);
        break;
      case 'z':
        face_direction = atoi(optarg);
        break;
      default:
        printf("Error: unknown option %c\n", (char)opt);
        exit(-1);
    }
  }
  assert(bindex_num >= 1);

  CODE *initial_data[MAX_BINDEX_NUM];
  if (!strlen(DATA_PATH)) {
    printf("initing data by random\n");
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      initial_data[bindex_id] = (CODE *)malloc(N * sizeof(CODE));
      CODE *data = initial_data[bindex_id];
      for (int i = 0; i < N; i++) {
        data[i] = i % (long(data_range_list[bindex_id]) + 1);
      }
      random_shuffle(data, data + N);
    }
  } else {
    getDataFromFile(DATA_PATH, initial_data, bindex_num);
  }

  size_t avail_init_gpu_mem, total_gpu_mem;
  size_t avail_curr_gpu_mem;
  cudaMemGetInfo( &avail_init_gpu_mem, &total_gpu_mem );

  size_t init_mem = memory_used_by_process();

  CODE *sorted_data[MAX_BINDEX_NUM];
  POSTYPE *sorted_pos[MAX_BINDEX_NUM];
  
#ifdef TEST
  for (int i = 0; i < N; i++) {
    printf("%3d: (%u, %u, %u)\n", i, initial_data[0][i], initial_data[1][i], initial_data[2][i]);
  }
#endif

#if ENCODE == 1
  timer.commonGetStartTime(20);
  #if TPCH == 1
  normalEncode(initial_data, bindex_num, 2, 4294967294, N, sorted_pos, sorted_data); // encoding for TPCH
  #else
  // When obtaining the memory for the mapping table, sorted_pos and sorted_data should be freed.
  normalEncode(initial_data, bindex_num, 0, N, N, sorted_pos, sorted_data); // encoding to range [encode_min, encode_max)
  #endif
  timer.commonGetEndTime(20);
#endif
  std::cerr << "Uniform Encoding done." << std::endl;
  std::cerr << "[Time] Uniform Encoding: " << timer.time[20] << std::endl;

#ifdef TEST
  for (int i = 0; i < N; i++) {
    printf("%3d: (%u, %u, %u)\n", i, initial_data[0][i], initial_data[1][i], initial_data[2][i]);
  }
  return 0;
#endif

  // init bindex
  BinDex *bindexs[MAX_BINDEX_NUM];
  timer.commonGetStartTime(21);
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    std::cerr << "Build the bindex structure " << bindex_id << "..." << std::endl;
    CODE *data = initial_data[bindex_id];
    bindexs[bindex_id] = (BinDex *)malloc(sizeof(BinDex));
    init_bindex_in_GPU(bindexs[bindex_id], data, N, bindex_id, sorted_pos[bindex_id], sorted_data[bindex_id]);
  }
  timer.commonGetEndTime(21);
  std::cerr << "Build Sieve Bit Vector done." << std::endl;
  std::cerr << "[Time] Build Sieve Bit Vector: " << timer.time[21] << std::endl;

  if (!malloc_trim(0)) {
    std::cerr << "malloc_trim failed!" << std::endl;
    exit(-1);
  }
  size_t used_mem = memory_used_by_process() - init_mem;
  cout << "[Mem] Uniform Encoding CPU memery used(MB): " << 1.0 * used_mem / (1 << 10) << std::endl;
  
  cudaMemGetInfo( &avail_curr_gpu_mem, &total_gpu_mem );
  size_t sieve_used = avail_init_gpu_mem - avail_curr_gpu_mem;
  cout << "[Mem] Sieve Bit Vector used(MB): " << 1.0 * sieve_used / (1 << 20) << std::endl;

  CODE *range = (CODE *) malloc(sizeof(CODE) * 6);
  for (int i = 0; i < bindex_num; i++) {
    range[i * 2] = bindexs[i]->data_min;
    range[i * 2 + 1] = bindexs[i]->data_max;
    
    if (min_val > bindexs[i]->data_min) min_val = bindexs[i]->data_min;
    if (max_val < bindexs[i]->data_max) max_val = bindexs[i]->data_max;
  }
  if (with_refine) {
    timer.commonGetStartTime(22);
    initializeOptix(initial_data, N, density_width, density_height, 3, range, cube_width);
    timer.commonGetEndTime(22);
    std::cerr << "Initialize RT done." << std::endl;
    std::cerr << "[Time] Initialize RT: " << timer.time[22] << std::endl;
  }
  cudaMemGetInfo( &avail_curr_gpu_mem, &total_gpu_mem );
  size_t rt_used = avail_init_gpu_mem - avail_curr_gpu_mem - sieve_used;
  cout << "[Mem] RT used(MB): " << 1.0 * rt_used / (1 << 20) << std::endl;
  
  if (default_ray_length == -2) {
    default_ray_length = (max_val - min_val) / default_ray_segment_num;
  }

  printf("BinDex scan...\n");

  // init result in CPU memory
  BITS *bitmap[MAX_BINDEX_NUM]; // result
  int bitmap_len;
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    bitmap_len = bits_num_needed(bindexs[bindex_id]->length);
    bitmap[bindex_id] = (BITS *)aligned_alloc(SIMD_ALIGEN, bitmap_len * sizeof(BITS));
    memset_mt(bitmap[bindex_id], 0xFF, bitmap_len);
  }
  bitmap[bindex_num] = (BITS *)aligned_alloc(SIMD_ALIGEN, bitmap_len * sizeof(BITS));
  memset_mt(bitmap[bindex_num], 0xFF, bitmap_len); 

  // init result in GPU memory
  BITS *dev_bitmap[MAX_BINDEX_NUM];
  cudaError_t cudaStatus;
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    bitmap_len = bits_num_needed(bindexs[bindex_id]->length);
    cudaStatus = cudaMalloc((void**)&(dev_bitmap[bindex_id]), bitmap_len * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed when init dev bitmap!");
        exit(-1);
    }
    cudaMemset(dev_bitmap[bindex_id], 0xFF, bitmap_len * sizeof(BITS));
  }
  cudaStatus = cudaMalloc((void**)&(dev_bitmap[bindex_num]), bitmap_len * sizeof(BITS));
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed when init dev bitmap!");
      exit(-1);
  }
  cudaMemset(dev_bitmap[bindex_num], 0xFF, bitmap_len * sizeof(BITS));
  cudaMemGetInfo( &avail_curr_gpu_mem, &total_gpu_mem );
  size_t result_bv_used = avail_init_gpu_mem - avail_curr_gpu_mem - sieve_used - rt_used;
  cout << "[Mem] Result Bit Vector used(MB): " << 1.0 * result_bv_used / (1 << 20) << std::endl;

  vector<CODE> target_lower;
  vector<CODE> target_upper; 
  vector<string> search_cmd;

  if (!READ_QUERIES_FROM_FILE) {  // generate queries
    if (RANGE_QUERY) {
      generate_range_queries(target_lower, target_upper, search_cmd, bindex_num, range);
    } else {
      generate_point_queries(target_lower, search_cmd, bindex_num);
    }
  } else {            // read queries from file
    ifstream fin(SCAN_FILE);
    if (!fin.is_open()) {
      std::cerr << "Fail to open FILE " << SCAN_FILE << std::endl;
      exit(-1);
    }
    printf("[LOG] Number of queries: %d\n", NUM_QUERIES);
    target_lower.resize(NUM_QUERIES * bindex_num);
    target_upper.resize(NUM_QUERIES * bindex_num);
    search_cmd.resize(NUM_QUERIES * bindex_num);
    for (int i = 0; i < NUM_QUERIES; i++) {
      for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
        cout << "input [operator] [target_l] [target_r] (" << bindex_id + 1 << "/" << bindex_num << ")" << std::endl;
        string input;
        
        getline(fin, input);
        cout << input << std::endl;
        std::vector<std::string> cmds = stringSplit(input, ' ');
        if (cmds[0] == "exit") exit(0);
        search_cmd[i * bindex_num + bindex_id] = cmds[0];
        if (cmds.size() > 1) {
if (ENCODE) {
          timer.commonGetStartTime(16);
          target_lower[i * bindex_num + bindex_id] = encodeQuery(bindex_id, get_target_numbers(cmds[1])[0]);
          timer.commonGetEndTime(16);
          printf("[ENCODE] %u", target_lower[i * bindex_num + bindex_id]);
} else {
          target_lower[i * bindex_num + bindex_id] = get_target_numbers(cmds[1])[0];
}
        }
        if (cmds.size() > 2) {
if (ENCODE) {
          timer.commonGetStartTime(16);
          target_upper[i * bindex_num + bindex_id] = encodeQuery(bindex_id, get_target_numbers(cmds[2])[0]);
          timer.commonGetEndTime(16);
          printf(" %u", target_upper[i * bindex_num + bindex_id]);
} else {
          target_upper[i * bindex_num + bindex_id] = get_target_numbers(cmds[2])[0];
        }
}
        if (ENCODE) {
        printf("\n");
        }
      }
    }
  }
  
  timer.time[16] /= NUM_QUERIES; // Average time to encode a query
  timer.showMajorTime();
  std::cerr << "Generate queries done." << std::endl;

  for (int i = 0; i < NUM_QUERIES; i++) {
    vector<CODE> target_l[bindex_num];
    vector<CODE> target_r[bindex_num];
    CODE target_l_new[bindex_num];
    CODE target_r_new[bindex_num];
    for (int j = 0; j < bindex_num; j++) {
      target_l[j].push_back(target_lower[i * bindex_num + j]);
      target_l_new[j] = target_l[j][0];
      if (search_cmd[i * bindex_num + j] == "bt") {
        target_r[j].push_back(target_upper[i * bindex_num + j]);
        target_r_new[j] = target_r[j][0];
      }
    }
    
    // clean up refine slot
    scan_refine_in_position = 0;
    for(int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      scan_selected_compares[bindex_id][0] = 0;
      scan_selected_compares[bindex_id][1] = 0;
      scan_max_compares[bindex_id][0] = 0;
      scan_max_compares[bindex_id][1] = 0;
      scan_skip_other_face[bindex_id] = false;
      scan_skip_this_face[bindex_id] = false;
      scan_inverse_this_face[bindex_id] = false;
    }
    scan_skip_refine = false;

    timer.commonGetStartTime(11);
    // special scan for = (eq) operator
    bool eq_scan = false;
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      if (search_cmd[bindex_id] == "eq") {
        cudaMemset(dev_bitmap[0], 0, bitmap_len * sizeof(BITS));
        special_eq_scan(target_l_new, target_r_new, bindexs, dev_bitmap[0], bindex_num, search_cmd.data() + i * bindex_num);
        eq_scan = true;
        break;
      }
    }
    
    if (!eq_scan){
      assert(THREAD_NUM >= bindex_num);
      std::thread threads[THREAD_NUM];
      timer.commonGetStartTime(4);
      for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
        // scan multi-thread with GPU
        threads[bindex_id] = std::thread(scan_multithread_withGPU, 
                                          &(target_l[bindex_id]), 
                                          &(target_r[bindex_id]), 
                                          target1, 
                                          target2, 
                                          search_cmd[i * bindex_num + bindex_id], 
                                          bindexs[bindex_id], 
                                          dev_bitmap[bindex_id],
                                          bindex_id
        );
      }
      for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) threads[bindex_id].join();
      timer.commonGetEndTime(4);

      // merge should be done before refine now since new refine relies on dev_bitmap[0]
      merge_with_GPU(dev_bitmap[0], dev_bitmap, bindex_num, bindexs[0]->length);

      if (REDUCED_SCANNING) {
        double rc = 1.0;
        for (int i = 0; i < bindex_num; i++) {
          if (scan_selected_compares[i][1] == 0) {
            rc = 0;
            break;
          }
          rc *= 1.0 * scan_selected_compares[i][0] / scan_selected_compares[i][1];
          printf("scan_selected_compares[%d] = %u %u\n", i, scan_selected_compares[i][0], scan_selected_compares[i][1]);
        }
        printf("[LOG] REDUCED_SCANNING: %lf\n", rc);
        continue;
      }
      
#if DEBUG_INFO == 1
      for (int i = 0; i < 6; i++) {
        printf("%u < x < %u\n", scan_selected_compares[i][0], scan_selected_compares[i][1]);
      }
#endif

      if (with_refine) {
        // GPU refine here
        std::thread refine_thread = std::thread(refine_with_GPU, bindexs, dev_bitmap[0], bindex_num);
        refine_thread.join();
      }
    }
    timer.commonGetEndTime(11);

    if (!with_refine) {
      timer.showTime();
      timer.clear();
      continue;
    }

    // transfer GPU result back to memory
    BITS *h_result;
    cudaStatus =  cudaMallocHost((void**)&(h_result), bitmap_len * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMallocHost failed when init h_result!");
      exit(-1);
    }
    timer.commonGetStartTime(12);
    cudaStatus = cudaMemcpy(h_result, dev_bitmap[0], bits_num_needed(bindexs[0]->length) * sizeof(BITS), cudaMemcpyDeviceToHost); // only transfer bindex[0] here. may have some problems.
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "[ERROR]Result transfer, cudaMemcpy failed!");
        exit(-1);
    }
    timer.commonGetEndTime(12);
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
      raw_scan_entry(&(target_l[bindex_id]), 
                     &(target_r[bindex_id]), 
                     search_cmd[i * bindex_num + bindex_id], 
                     bindexs[bindex_id], 
                     check_bitmap[bindex_id],
                     check_bitmap[0],
                     initial_data[bindex_id]
      );
    }

    compare_bitmap(check_bitmap[0], h_result, bindexs[0]->length, initial_data, bindex_num);
    printf("[CHECK]check final result done.\n\n");

    cudaFreeHost(h_result);
  }

  // clean jobs
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    free(initial_data[bindex_id]);
    free_bindex(bindexs[bindex_id]);
    free(bitmap[bindex_id]);
  }
  return 0;
}
