// time count part
// semaphore
#include <mutex>
#include <sys/time.h>
#include <iostream>

using namespace std;

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

  void quickGetStartTime(int *timeArray, int timeId) {
    struct timeval t1;
    gettimeofday(&t1, NULL);
    timeArray[timeId] -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }

  void quickGetEndTime(int *timeArray, int timeId) {
    struct timeval t1;
    gettimeofday(&t1, NULL);
    timeArray[timeId] += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }

  double quickGetStartTime(double time) {
    struct timeval t1;
    gettimeofday(&t1, NULL);
    time -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
    return time;
  }

  double quickGetEndTime(double time) {
    struct timeval t1;
    gettimeofday(&t1, NULL);
    time += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
    return time;
  }
  
  void showTime() {
    cout << endl;
    cout << "###########   Time  ##########" << endl;
    cout << "[Time] query to ray (ray building): ";
    cout << time[23] << " ms" << endl;
    
    // cout << "[Time] encode data: ";
    // cout << time[1] << " ms" << endl;
    // cout << "[Time] GPU bits and: ";
    // cout << time[2] << endl;
    // cout << "[Time] GPU bits copy: ";
    // cout << time[3] << endl;
    cout << "[Time] data sieving: ";
    cout << time[4] << " ms" << endl;
    cout << "[Time] merge: "; // merge 3 sieving bit vectors
    cout << time[15] << " ms" << endl;
    cout << "[Time] refine: ";
    cout << time[13] << " ms" << endl;    
    // cout << "[Time] predicate malloc: ";
    // cout << time[18] << " ms" << endl;
    cout << "[Time] total scan: ";
    cout << time[11] << " ms" << endl;
    cout << "[Time] transfer result back to memory: ";
    cout << time[12] << " ms" << endl;
    cout << "[Time] total time: " << time[11] + time[12] << " ms" << endl;
    // add new time counter here
    cout << "##############################" << endl;
    cout << endl;
  }

  void showMajorTime() {
    cout << "###########   Major Time  ##########" << endl;
    cout << "[Time] uniform encoding: ";
    cout << time[20] << " ms" << endl;
    cout << "[Time] build sieve bit vector: ";
    cout << time[21] << " ms" << endl;
    cout << "[Time] initialize RT: ";
    cout << time[22] << " ms" << endl;
    cout << "[Time] encode query: ";
    cout << time[16] << " ms" << endl;
    cout << "[Time] transfer sieve bit vector: ";
    cout << time[24] << " ms" << endl;
    cout << "####################################" << endl;
  }
};
