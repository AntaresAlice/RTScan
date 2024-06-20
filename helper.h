#ifndef HELPER_H_
#define HELPER_H_
#include <vector>
#include <string>
#include <sys/time.h>
#include <regex>
#include <thread>
using namespace std;

vector<string> stringSplit(const string& str, char delim) {
    string s;
    s.append(1, delim);
    regex reg(s);
    vector<string> elems(sregex_token_iterator(str.begin(), str.end(), reg, -1),
                                   sregex_token_iterator());
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

size_t memory_used_by_process() {
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];
  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      int len = strlen(line);
      const char* p = line;
      for (; isdigit(*p) == false; ++p) {}
      line[len - 3] = 0;
      result = atoi(p);
      break;
    }
  }
  fclose(file);
  return result;  // KB
}

void memset_numa0(uint32_t *p, int val, int n, int t_id, int thread_num, int simd_aligen) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(t_id * 2, &mask);
  if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
    fprintf(stderr, "set thread affinity failed\n");
  }
  int avg_workload = (n / (thread_num * simd_aligen)) * simd_aligen;
  int start = t_id * avg_workload;
  int end = t_id == (thread_num - 1) ? n : start + avg_workload;
  memset(p + start, val, (end - start) * sizeof(uint32_t));
}

void memset_mt(uint32_t *p, int val, int n, int thread_num, int simd_aligen) {
  thread threads[thread_num];
  for (int t_id = 0; t_id < thread_num; t_id++) {
    threads[t_id] = thread(memset_numa0, p, val, n, t_id, thread_num, simd_aligen);
  }
  for (int t_id = 0; t_id < thread_num; t_id++) {
    threads[t_id].join();
  }
}

vector<uint32_t> get_target_numbers(string s) {
  stringstream ss(s);
  string value;
  vector<uint32_t> result;
  while (getline(ss, value, ',')) {
    result.push_back((uint32_t)stod(value));
  }
  return result;
}
#endif