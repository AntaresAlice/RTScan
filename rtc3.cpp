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

int default_ray_length = 48000000;
int default_ray_mode = 0; // 0 for continuous ray, 1 for ray with space, 2 for ray as point

CODE min_val = UINT32_MAX;
CODE max_val = 0;

int NUM_QUERIES;
int RANGE_QUERY = 1;
uint32_t data_range_list[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
int direction = 0; // 0:X, 1:Y, 2:Z
uint32_t cube_width = 0;
bool read_query_from_file = true;
bool switch_tpch = false;

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
      // Malloc 2 times of space, prepared for future appending
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
      if (fread(data, sizeof(uint32_t), N, fp) == 0) {
        printf("init_data_from_file: fread faild.\n");
        exit(-1);
      }
      for (int i = 0; i < N; i++) {
        data[i] = data[i] & data_range_list[bindex_id];
      }
    } else {
      printf("init_data_from_file: CODE_WIDTH != 8/16/32.\n");
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

void raw_scan(BinDex *bindex, BITS *bitmap, CODE target1, CODE target2, OPERATOR OP, CODE *raw_data, BITS* compare_bitmap = NULL) {
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
    }
    else if (search_cmd[bindex_id] == "lt") {
      compares[bindex_id][0] = bindexs[bindex_id]->data_min - 1.0;
      compares[bindex_id][1] = double(target_l[bindex_id]);
      if (compares[bindex_id][0] > compares[bindex_id][1]) {
        swap(compares[bindex_id][0],compares[bindex_id][1]);
      }
    }
    else {
      // printf("[ERROR] %s not support yet!\n", search_cmd[bindex_id]);
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
  // default_ray_length = -1;
  refineWithOptixRTc3(dev_bitmap, dev_predicate, bindex_num, default_ray_length, default_ray_segment_num, false, direction, default_ray_mode);
  if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
}

int get_wide_face(double **compares, int bindex_num)
{
  double min_distance = compares[0][1] - compares[0][0];
  int wide_face_id = 0;
  for (int bindex_id = 1; bindex_id < bindex_num; bindex_id++) {
    if (compares[bindex_id][1] - compares[bindex_id][0] < min_distance) {
      min_distance = compares[bindex_id][1] - compares[bindex_id][0];
      wide_face_id = bindex_id;
    }
  }
  return wide_face_id;
}

int get_refine_space_side(double compares[][2], int bindex_num, int face_direction) {
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
  int face_direction = 1;  // 0 = all wide, 1 = all narrow
  bool adjust_ray_num = false; // switch for fixed ray number: `true` for `fixed`
  int default_ray_total_num = 20000;
  int ray_length = default_ray_length; // set to -1 so rt will use default segmentnum set

  double **compares = (double **)malloc(bindex_num * sizeof(double *));
  double *dev_predicate = (double *)malloc(bindex_num * 2 * sizeof(double));
  for (int i = 0; i < bindex_num; i++) {
    compares[i] = &(dev_predicate[i * 2]);
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

      int direction = get_wide_face(compares, bindex_num); // from wide face
      if (face_direction == 1) {
        for(int i = 2; i >= 0; i--) {
          if (i != direction) {
            direction = i;
            break;
          }
        } // from narrow face
      }

      // calculate ray_segment_num
      if (adjust_ray_num) default_ray_segment_num = calculate_ray_segment_num(direction, dev_predicate, bindexs, default_ray_total_num);
      
      for (int i = 0; i < bindex_num; i++) {
        if (compares[i][0] == compares[i][1]) {
          if(DEBUG_INFO) printf("[INFO] %d face scan skipped for the same compares[0] and compares[1].\n",bindex_id);
          if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
          return;
        }
      }

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
      refineWithOptixRTc3(dev_bitmap, dev_predicate, bindex_num, ray_length, default_ray_segment_num, false, direction, default_ray_mode);
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

    int direction = get_wide_face(compares, bindex_num); // from wide face
    if (face_direction == 1) {
        for(int i = 2; i >= 0; i--) {
          if (i != direction) {
            direction = i;
            break;
          }
        } // from narrow face
      }

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
    refineWithOptixRTc3(dev_bitmap, dev_predicate, bindex_num, ray_length, default_ray_segment_num, inverse, direction, default_ray_mode);
    
    max_MS_face_count += 1;
  }
  if(DEBUG_INFO) printf("total selectivity: %f\n", selectivity);
  if(DEBUG_TIME_COUNT) timer.commonGetEndTime(13);
  return;
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

// Only support `lt` now.
void generate_range_queries(vector<CODE> &target_lower, 
                            vector<CODE> &target_upper,
                            vector<string> &search_cmd,
                            int column_num,
                            CODE *range) {
// #if DISTRITION == 0                              
//   CODE span = UINT32_MAX / (NUM_QUERIES + 1);
// #else
//   CODE span = (max_val - min_val + NUM_QUERIES - 1) / NUM_QUERIES;
// #endif

//   target_lower.resize(NUM_QUERIES * column_num);
//   search_cmd.resize(NUM_QUERIES * column_num);
//   for (int i = 0; i < NUM_QUERIES; i++) {
//     for (int column_id = 0; column_id < column_num; column_id++) {
//       target_lower[i * column_num + column_id] = (i + 1) * span;
//       search_cmd[i * column_num + column_id] = "lt";
//     }
//   }

  // CODE span = (max_val - min_val + (NUM_QUERIES - 1) - 1) / (NUM_QUERIES - 1);
  // CODE short_span = ((1 << RAY_INTERVAL) + (NUM_QUERIES - 1) - 1) / (NUM_QUERIES - 1);

  // target_lower.resize(NUM_QUERIES * column_num);
  // search_cmd.resize(NUM_QUERIES * column_num);
  // for (int i = 0; i < NUM_QUERIES; i++) {
  //   for (int column_id = 0; column_id < column_num; column_id++) {
  //     if (column_id == 0) {
  //       target_lower[i * column_num + column_id] = i * span;
  //     } else {
  //       target_lower[i * column_num + column_id] = i * short_span;
  //     }
  //     search_cmd[i * column_num + column_id] = "lt";
  //   }
  // }

  double span = 1.0 * (max_val - min_val) / (NUM_QUERIES - 1);
  target_lower.resize(NUM_QUERIES * column_num);
  search_cmd.resize(NUM_QUERIES * column_num);
  for (int i = 0; i < NUM_QUERIES; i++) {
    for (int column_id = 0; column_id < column_num; column_id++) {
      target_lower[i * column_num + column_id] = CODE(i * span);
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

  for (int i = 0; i < NUM_QUERIES; i++) {
    for (int column_id = 0; column_id < column_num; column_id++) {
      printf("%u ", target_lower[i * column_num + column_id]);
    }
    printf("\n");
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

int main(int argc, char *argv[]) {
  printf("N = %d\n", N);
  printf("DISTRIBUTION: %d\n", DISTRIBUTION);
  
  char opt;
  int selectivity;
  char DATA_PATH[256] = "\0";
  char SCAN_FILE[256] = "\0";
  int bindex_num = 3;

  density_width = 1200;
  density_height = 1200; // maximum ray-width and maximum ray-height 
  default_ray_segment_num = 64;

  while ((opt = getopt(argc, argv, "hf:a:b:w:m:s:t:p:q:g:v:c:d:z:e:")) != -1) {
    switch (opt) {
      case 'h':
        printf(
            "Usage: %s \n"
            "[-f <input-file>]\n"
            "[-w <ray-range-width>] [-m <ray-range-height>]\n"
            "[-s <ray-segment-num>]\n"
            "[-p <scan-predicate-file>]\n"
            "[-a <ray-length>]\n"
            "[-q <query-num>]\n"
            "[-g <range-query>]\n"
            "[-z <read-query-from-file>]\n"
            "[-e <ray-mode>]",
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
      case 't':
        switch_tpch = atoi(optarg);
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
      case 'q':
        NUM_QUERIES = atoi(optarg);
        break;
      case 'g':
        RANGE_QUERY = atoi(optarg);
        break;
      case 'c':
        cube_width = atol(optarg);
        break;
      case 'd':
        direction = atoi(optarg);
        break;
      case 'v':
        sscanf(optarg, "%u,%u,%u", &data_range_list[0], &data_range_list[1], &data_range_list[2]);
        break;
      case 'z':
        read_query_from_file = atoi(optarg);
        break;
      case 'e':
        default_ray_mode = atoi(optarg);
        break;
      default:
        printf("Error: unknown option %c\n", (char)opt);
        exit(-1);
    }
  }
  assert(NUM_QUERIES);
  assert(bindex_num >= 1);

  // initial data
  CODE *initial_data[MAX_BINDEX_NUM];

  if (!strlen(DATA_PATH)) {
    printf("initing data by random\n");
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      initial_data[bindex_id] = (CODE *)malloc(N * sizeof(CODE));
      CODE *data = initial_data[bindex_id];
      for (int i = 0; i < N; i++) {
        data[i] = i % (data_range_list[bindex_id] + 1);
      }
      random_shuffle(data, data + N);
    }
  } else {
    getDataFromFile(DATA_PATH, initial_data, bindex_num);
  }

  // remap/encode initialdata
  if (ENCODE) {
    printf("[+] remapping initial data...\n");
    if (switch_tpch) {
      normalEncode(initial_data, bindex_num, 2, 4294967294, N);
    } else {
      normalEncode(initial_data, bindex_num, 0, N, N); // encoding to range [encode_min, encode_max)
    }
    malloc_trim(0);
    printf("[+] remap initial data done.\n");
  }

  // init bindex
  BinDex *bindexs[MAX_BINDEX_NUM];
  CODE *raw_datas[MAX_BINDEX_NUM];
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    printf("Build the bindex structure %d...\n", bindex_id);
    CODE *data = initial_data[bindex_id];
    bindexs[bindex_id] = (BinDex *)malloc(sizeof(BinDex));
    raw_datas[bindex_id] = (CODE *)malloc(2 * N * sizeof(CODE));          // Store unsorted raw data in bindex
    if (DEBUG_TIME_COUNT) timer.commonGetStartTime(0);
    init_bindex_in_GPU(bindexs[bindex_id], data, N, raw_datas[bindex_id]);
    if (DEBUG_TIME_COUNT) timer.commonGetEndTime(0);
    printf("\n");
  }

  CODE *range = (CODE *) malloc(sizeof(CODE) * 6);
  for (int i = 0; i < bindex_num; i++) {
    range[i * 2] = bindexs[i]->data_min;
    range[i * 2 + 1] = bindexs[i]->data_max;
    
    if (min_val > bindexs[i]->data_min) min_val = bindexs[i]->data_min;
    if (max_val < bindexs[i]->data_max) max_val = bindexs[i]->data_max;
  }
  initializeOptixRTc3(initial_data, N, density_width, density_height, 3, range, cube_width, direction);

  printf("BinDex scan...\n");

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
    cudaMemset(dev_bitmap[bindex_id], 0, bitmap_len * sizeof(BITS));
  }
  cudaStatus = cudaMalloc((void**)&(dev_bitmap[bindex_num]), bitmap_len * sizeof(BITS));
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed when init dev bitmap!");
      exit(-1);
  }
  cudaMemset(dev_bitmap[bindex_num], 0, bitmap_len * sizeof(BITS));

  vector<CODE> target_lower;
  vector<CODE> target_upper;
  vector<std::string> search_cmd;

  if (!read_query_from_file) {  // generate queries
    if (RANGE_QUERY) {
      generate_range_queries(target_lower, target_upper, search_cmd, bindex_num, range);
    } else {
      generate_point_queries(target_lower, search_cmd, bindex_num);
    }
  } else {            // read queries from file
    ifstream fin;
    if (!strlen(SCAN_FILE)) {
      fin.open("test/scan_cmd.txt");
    } else {
      fin.open(SCAN_FILE);
    }

    target_lower.resize(NUM_QUERIES * bindex_num);
    target_upper.resize(NUM_QUERIES * bindex_num);
    search_cmd.resize(NUM_QUERIES * bindex_num);
    for (int i = 0; i < NUM_QUERIES; i++) {
      for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
        cout << "input [operator] [target_l] [target_r] (" << bindex_id + 1 << "/" << bindex_num << ")" << endl;
        string input;
        
        getline(fin, input);
        cout << input << endl;
        std::vector<std::string> cmds = stringSplit(input, ' ');
        if (cmds[0] == "exit")
          exit(0);
        search_cmd[i * bindex_num + bindex_id] = cmds[0];
        if (cmds.size() > 1) {
          target_lower[i * bindex_num + bindex_id] = get_target_numbers(cmds[1])[0];
        }
        if (cmds.size() > 2) {
          target_upper[i * bindex_num + bindex_id] = get_target_numbers(cmds[2])[0];
        }
      }
    }
  }

  double *predicate = (double *)malloc(6 * sizeof(double));

  printf("NUM_QUERIES: %d\n", NUM_QUERIES);
  for (int i = 0; i < NUM_QUERIES; i++) {
    cudaMemset(dev_bitmap[0], 0, bitmap_len * sizeof(BITS));
    if (switch_tpch) {
      predicate[0] = target_lower[i * bindex_num + 0];
      predicate[1] = target_upper[i * bindex_num + 0];
      
      predicate[2] = target_lower[i * bindex_num + 1];
      predicate[3] = target_upper[i * bindex_num + 1];
      
      predicate[4] = target_lower[i * bindex_num + 2];
      predicate[5] = target_upper[i * bindex_num + 2];
    } else {
      predicate[0] = double(range[0]) - 1;
      predicate[1] = target_lower[i * bindex_num + 0];
      
      predicate[2] = double(range[2]) - 1;
      predicate[3] = target_lower[i * bindex_num + 1];
      
      predicate[4] = double(range[4]) - 1;
      predicate[5] = target_lower[i * bindex_num + 2];
    }
    for (int pi = 0; pi < 6; pi++) {
      printf("%.0lf ", predicate[pi]);
    }
    printf("\n");

    timer.commonGetStartTime(13);
    if (switch_tpch) {
      double compares[3][2];
      for (int ci = 0; ci < 3; ci++) {
        compares[ci][0] = predicate[ci * 2];
        compares[ci][1] = predicate[ci * 2 + 1];
      }
      direction = get_refine_space_side(compares, bindex_num, 1);
    }
    refineWithOptixRTc3(dev_bitmap[0], predicate, 3, default_ray_length, default_ray_segment_num, false, direction, default_ray_mode);
    timer.commonGetEndTime(13);
    timer.time[11] = timer.time[13];

    vector<CODE> target_l[3], target_r[3];
    if (switch_tpch) {
      target_l[0].push_back(predicate[0]);
      target_r[0].push_back(predicate[1]);
      target_l[1].push_back(predicate[2]);
      target_r[1].push_back(predicate[3]);
      target_l[2].push_back(predicate[4]);
      target_r[2].push_back(predicate[5]);
    } else {
      target_l[0].push_back(predicate[1]);
      target_l[1].push_back(predicate[3]);
      target_l[2].push_back(predicate[5]);
    }

    BITS* h_result;
    cudaStatus = cudaMallocHost((void**)&(h_result), bitmap_len * sizeof(BITS));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMallocHost failed when init h_result!");
      exit(-1);
    }
    timer.commonGetStartTime(12);
    cudaStatus = cudaMemcpy(h_result, dev_bitmap[0], bits_num_needed(bindexs[0]->length) * sizeof(BITS), cudaMemcpyDeviceToHost);  // only transfer bindex[0] here. may have some problems.
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "[ERROR]Result transfer, cudaMemcpy failed!");
      exit(-1);
    }
    timer.commonGetEndTime(12);
    timer.showTime();
    timer.clear();

    // check jobs
    BITS* check_bitmap[MAX_BINDEX_NUM];
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      check_bitmap[bindex_id] = (BITS*)aligned_alloc(SIMD_ALIGEN, bitmap_len * sizeof(BITS));
      memset_mt(check_bitmap[bindex_id], 0x0, bitmap_len);
    }

    printf("[CHECK]check final result.\n");
    for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
      raw_scan_entry(&(target_l[bindex_id]),
                     &(target_r[bindex_id]),
                     search_cmd[i * bindex_num + bindex_id], 
                     bindexs[bindex_id],
                     check_bitmap[bindex_id],
                     check_bitmap[0],
                     raw_datas[bindex_id]);
    }

    compare_bitmap(check_bitmap[0], h_result, bindexs[0]->length, raw_datas, bindex_num);
    printf("[CHECK]check final result done.\n");

    cudaFreeHost(h_result);
  }

  // clean jobs
  for (int bindex_id = 0; bindex_id < bindex_num; bindex_id++) {
    free(initial_data[bindex_id]);
    free_bindex(bindexs[bindex_id], raw_datas[bindex_id]);
    free(bitmap[bindex_id]);
  }
  return 0;
}