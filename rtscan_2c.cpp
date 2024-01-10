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

int density_width;
int density_height;

int default_ray_segment_num;

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::string s;
    s.append(1, delim);
    std::regex reg(s);
    std::vector<std::string> elems(std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   std::sregex_token_iterator());
    return elems;
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

void init_bindex_in_GPU(BinDex *bindex, CODE *data, POSTYPE n, CODE *raw_data) {
  bindex->length = n;
  POSTYPE avgAreaSize = n / K;
  cudaError_t cudaStatus;

  POSTYPE areaStartIdx[K];
  CODE *data_sorted = (CODE *)malloc(n * sizeof(CODE));  // Sorted codes
  POSTYPE *pos = argsort(data, n);
  for (int i = 0; i < n; i++) {
    data_sorted[i] = data[pos[i]];
    raw_data[i] = data[i];
  }

  bindex->data_min = data_sorted[0];
  bindex->data_max = data_sorted[bindex->length - 1];

  printf("Bindex data min: %u  max: %u\n", bindex->data_min, bindex->data_max);

  bindex->areaStartValues[0] = data_sorted[0];
  areaStartIdx[0] = 0;

  for (int i = 1; i < K; i++) {
    bindex->areaStartValues[i] = data_sorted[i * avgAreaSize];
    int j = i * avgAreaSize;
    if (bindex->areaStartValues[i] == bindex->areaStartValues[i - 1]) {
      areaStartIdx[i] = j;
    } else {
      // To find the first element which is less than startValue
      while (data_sorted[j] == bindex->areaStartValues[i]) {
        j--;
      }
      areaStartIdx[i] = j + 1;
    }
  }
  
  std::thread threads[THREAD_NUM];

  // Build the filterVectors
  // Now we build them in CPU memory and then copy them to GPU memory
  for (int k = 0; k * THREAD_NUM < K - 1; k++) {
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      bindex->filterVectors[k * THREAD_NUM + j] =
          (BITS *)aligned_alloc(SIMD_ALIGEN, bits_num_needed(n) * sizeof(BITS));
      threads[j] = std::thread(set_fv_val_less, bindex->filterVectors[k * THREAD_NUM + j], raw_data,
                               bindex->areaStartValues[k * THREAD_NUM + j + 1], n);
    }
    for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < (K - 1); j++) {
      threads[j].join();
    }
  }

  for (int i = 0; i < K - 1; i++) {
    cudaStatus = cudaMalloc((void**)&(bindex->filterVectorsInGPU[i]), bits_num_needed(n) * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(-1);
    }

    cudaStatus = cudaMemcpy(bindex->filterVectorsInGPU[i], bindex->filterVectors[i], bits_num_needed(n) * sizeof(BITS), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        exit(-1);
    }
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

  // handle some boundary problems: <0, ==0, K-1, >K-1
  // <0: set all bits to 0. should not call rt. interrupt other procedures now. should cancel merge as well?
  // ==0: set all bits to 0. just scan one face: SC[this bindex] x MC[other bindexs].
  // K-1: area_idx = K - 1 may not cause problem now?
  // K: set all bits to 1. skip this face: SC[this bindex] x MC[other bindexs]

  if (area_idx < 0) {
    // set skip to true so refine thread will be informed to skip
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

  // set refine compares here
  scan_refine_mutex.lock();
  if (inverse) {
    scan_selected_compares[bindex_id][0] = compare;
    scan_selected_compares[bindex_id][1] = bindex->areaStartValues[area_idx + 1];
  } else {
    scan_selected_compares[bindex_id][0] = bindex->areaStartValues[area_idx];
    scan_selected_compares[bindex_id][1] = compare;
  }
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

    // cudaMemset(dev_bitmap, 0xFF, bitmap_len);

    return;
  }


  // set refine compares here
  // scan_refine_mutex.lock();
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

void free_bindex(BinDex *bindex, CODE *raw_data) {
  free(raw_data);
  for (int i = 0; i < K - 1; i++) {
    free(bindex->filterVectors[i]);
  }
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

void getDataFromFile(char *DATA_PATH, CODE **initial_data, int bindex_num) {
  FILE *fp;
  if (!(fp = fopen(DATA_PATH, "rb"))) {
    printf("init_data_from_file: fopen(%s) faild\n", DATA_PATH);
    exit(-1);
  }
  printf("initing data from %s\n", DATA_PATH);

  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    initial_data[bindex_id] = (CODE *)malloc(N * sizeof(CODE));
    CODE *data = initial_data[bindex_id];
    if (fread(data, sizeof(uint32_t), N, fp) == 0) {
      printf("init_data_from_file: fread faild.\n");
      exit(-1);
    }
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
      printf("[ERROR] check error in raw_data[%d]= ", i);
      if (bindex_num == 2) {
        printf(" %u / %u \n", raw_data[0][i], raw_data[1][i]);
      } else if (bindex_num == 3) {
        printf(" %u / %u / %u \n", raw_data[0][i], raw_data[1][i], raw_data[2][i]);
      } else {
        fprintf(stderr, "bindex_num is illegal!\n");
      }
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

    for (int i = 0; i < RUNS; i++) {
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

void scan_multithread_withGPU(std::vector<CODE> *target_l, std::vector<CODE> *target_r, std::string search_cmd, BinDex *bindex, BITS *bitmap, int bindex_id) {
  CODE target1, target2;
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
    }
    printf("\n");
  }
}

void refine_with_GPU(BinDex **bindexs, BITS *dev_bitmap, const int bindex_num) {
  double **compares = (double **)malloc(bindex_num * sizeof(double *));
  double *dev_predicate = (double *)malloc(bindex_num * 2 * sizeof(double));
  CODE *range = (CODE *) malloc(sizeof(CODE) * 6);
  for (int i = 0; i < bindex_num; i++) {
    compares[i] = &(dev_predicate[i * 2]);
    range[i * 2] = bindexs[i]->data_min;
    range[i * 2 + 1] = bindexs[i]->data_max;
  }
  
  if(DEBUG_TIME_COUNT) timer.commonGetStartTime(13);
  // if there is a compare totally out of boundary, refine procedure can be skipped
  if (scan_skip_refine) {
    if(DEBUG_INFO) printf("[INFO] Search out of boundary, skip all refine.\n");
    if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
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
          return;
        }
      }

      // add refine here
      // send compares, dev_bitmap, the result is in dev_bitmap
      if(DEBUG_INFO) {
        printf("[Prepared predicate]");
        for (int i = 0; i < 6; i++) {
            printf("%f ", dev_predicate[i]);
        }
        printf("\n");
        printf("[INFO] compares prepared.\n");
      }

      refineWithOptixRTScan_2c(dev_bitmap, dev_predicate, range, bindex_num, default_ray_segment_num, false);
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
        if (DEBUG_INFO)
          printf("[INFO] %d face scan skipped for the same compares[0] and compares[1].\n", bindex_id);
        mid_skip_flag = true;
        break;
      }
    }
    if (mid_skip_flag)
      continue;

    if(DEBUG_INFO) {
      for (int i = 0; i < bindex_num; i++) {
        face_selectivity *= double(compares[i][1] - compares[i][0]) / double(bindexs[i]->data_max - bindexs[i]->data_min);
      }
      selectivity += face_selectivity;
    }

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
      printf("[INFO] compares prepared.\n");
    }
    refineWithOptixRTScan_2c(dev_bitmap, dev_predicate, range, bindex_num, default_ray_segment_num, inverse);
    
    max_MS_face_count += 1;
  }
  if(DEBUG_INFO) printf("total selectivity: %f\n", selectivity);
  if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
  return;
}

void merge_with_GPU(BITS *merge_bitmap, BITS **dev_bitmaps, const int bindex_num, const int bindex_len)
{
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

  // int skip_num = 0;
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    if (!scan_skip_this_face[bindex_id]) {
      if (merge_bitmap != dev_bitmaps[bindex_id]) {
        GPUbitAndWithCuda(merge_bitmap, dev_bitmaps[bindex_id], bindex_len);
      }
    }
    else {
      if(DEBUG_INFO) printf("[INFO] %d face merge skipped.\n",bindex_id);
      // skip_num++;
    }
  }

  timer.commonGetEndTime(15);
  return;
}

int Rand(int i) { 
  return rand() % i;
}

int main(int argc, char *argv[]) {
  char opt;
  int selectivity;
  char DATA_PATH[256] = "\0";
  char SCAN_FILE[256] = "\0";
  char OPERATOR_TYPE[5];
  int bindex_num = 2;

  density_width = 400;
  density_height = 400; // maximum ray-width and maximum ray-height 
  default_ray_segment_num = 1;

  while ((opt = getopt(argc, argv, "hf:b:w:m:s:p:")) != -1) {
    switch (opt) {
      case 'h':
        printf(
            "Usage: %s \n"
            "[-f <input-file>]\n"
            "[-w <ray-range-width>] [-m <ray-range-height>]\n"
            "[-s <ray-segment-num>]\n"
            "[-p <scan-predicate-file>]\n",
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
      for (int i = 1; i <= N; i++) {
        data[i] = i;
      }
      random_shuffle(data, data + N, Rand);
    }
  } else {
    getDataFromFile(DATA_PATH, initial_data, bindex_num);
  }
  
  // init optixScan
  initializeOptixRTScan_2c(initial_data, N, density_width, density_height, bindex_num);

  // init bindex
  BinDex *bindexs[MAX_BINDEX_NUM];
  CODE *raw_datas[MAX_BINDEX_NUM];
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    // Build the bindex structure
    printf("Build the bindex structure %d...\n", bindex_id);
    CODE *data = initial_data[bindex_id];
    bindexs[bindex_id] = (BinDex *)malloc(sizeof(BinDex));
    raw_datas[bindex_id] = (CODE *)malloc(2 * N * sizeof(CODE));          // Store unsorted raw data in bindex
    if (DEBUG_TIME_COUNT) timer.commonGetStartTime(0);
    // PRINT_EXCECUTION_TIME("BinDex building", init_bindex(bindexs[bindex_id], data, N, raw_datas[bindex_id]));
    init_bindex_in_GPU(bindexs[bindex_id], data, N, raw_datas[bindex_id]);
    if (DEBUG_TIME_COUNT) timer.commonGetEndTime(0);
    printf("\n");
  }

  printf("\n");

  printf("BinDex scan...\n");

  // init result in CPU memory
  // result
  BITS *bitmap[MAX_BINDEX_NUM];
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

  // malloc dev bitmap for result after refine
  BITS *dev_bitmap_for_refined_result;
  cudaStatus = cudaMalloc((void**)&(dev_bitmap_for_refined_result), bitmap_len * sizeof(BITS));
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed when init dev bitmap!");
      exit(-1);
  }
  cudaMemset(dev_bitmap_for_refined_result, 0x00, bitmap_len * sizeof(BITS));

  ifstream fin;
  if (!strlen(SCAN_FILE)) {
    fin.open("test/scan_cmd2.txt");
  } else {
    fin.open(SCAN_FILE);
  }
  int toExit = 1;
  while(toExit != -1) {
    //  TODO: using target_vector to collect search target is out of date since we don't support cmd like 'lt 1 1000 2000' any more
    //  just change this to CODE target1 and CODE target2
    std::vector<CODE> target_l[MAX_BINDEX_NUM];
    std::vector<CODE> target_r[MAX_BINDEX_NUM]; 
    string search_cmd[MAX_BINDEX_NUM];


    // TODO: change this to a multi-thread, thread finishing first can refine result first

    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      cout << "input [operator] [target_l] [target_r] (" << bindex_id + 1 << "/" << bindex_num << ")" << endl;
      string input;

      getline(fin, input);
      cout << input << endl;
      std::vector<std::string> cmds = stringSplit(input, ' ');
      if (cmds[0] == "exit")
        exit(0);
      search_cmd[bindex_id] = cmds[0];
      if (cmds.size() > 1) {
        target_l[bindex_id] = get_target_numbers(cmds[1]);
      }
      if (cmds.size() > 2) {
        target_r[bindex_id] = get_target_numbers(cmds[2]);
      }
    }

    // clean up refine slot
    scan_refine_in_position = 0;
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
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
    assert(THREAD_NUM >= bindex_num);
    std::thread threads[THREAD_NUM];
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      threads[bindex_id] = std::thread(scan_multithread_withGPU, 
                                       &(target_l[bindex_id]), 
                                       &(target_r[bindex_id]), 
                                       search_cmd[bindex_id], 
                                       bindexs[bindex_id], 
                                       dev_bitmap[bindex_id],
                                       bindex_id
      );
    }
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) threads[bindex_id].join();
    // merge should be done before refine now since new refine relies on dev_bitmap[0]
    merge_with_GPU(dev_bitmap[0], dev_bitmap, bindex_num, bindexs[0]->length);
    
    if (DEBUG_INFO) {
      for (int i = 0; i < bindex_num; i++) {
        printf("%u < x < %u\n", scan_selected_compares[i][0], scan_selected_compares[i][1]);
      }
    }
    refine_with_GPU(bindexs, dev_bitmap[0], bindex_num); 

    timer.commonGetEndTime(11);
    // transfer GPU result back to memory
    BITS *h_result;
    cudaStatus =  cudaMallocHost((void**)&(h_result), bitmap_len * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "[ERROR]cudaMallocHost failed when init h_result!");
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

    /// check final result 
    printf("[CHECK]check final result.\n");
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      raw_scan_entry(
                      &(target_l[bindex_id]), 
                      &(target_r[bindex_id]), 
                      search_cmd[bindex_id], 
                      bindexs[bindex_id], 
                      check_bitmap[bindex_id],
                      check_bitmap[0],
                      raw_datas[bindex_id]
      );
    }

    compare_bitmap(check_bitmap[0], h_result, bindexs[0]->length, raw_datas, bindex_num);
    printf("[CHECK]check final result done.\n\n");

    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      free(check_bitmap[bindex_id]);
    }
    cudaStatus = cudaFreeHost(h_result);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "[ERROR]Failed to free h_result!");
      exit(-1);
    }
  }

  // clean jobs
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    free(initial_data[bindex_id]);
    free_bindex(bindexs[bindex_id], raw_datas[bindex_id]);
    bindexs[bindex_id] = nullptr;
    raw_datas[bindex_id] = nullptr;
    free(bitmap[bindex_id]);
  }
  return 0;
}