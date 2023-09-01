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
    cout << "[Time] build bindex: ";
    cout << time[0] << " ms" << endl;
    cout << "[Time] encode data: ";
    cout << time[1] << " ms" << endl;
    cout << "[Time] GPU bits and: ";
    cout << fixed << time[2] << endl;
    cout << "[Time] GPU bits copy: ";
    cout << fixed << time[3] << endl;
    cout << "[Time] data sieving: ";
    cout << time[4] << " ms" << endl;
    cout << "[Time] first merge: "; // 将 3 个 draft vector 合并
    cout << time[15] << " ms" << endl;
    cout << "[Time] refine: ";
    cout << time[13] << " ms" << endl;    
    cout << "[Time] predicate malloc: ";
    cout << time[18] << " ms" << endl;
    cout << "[Time] encode query: ";
    cout << time[16] << " ms" << endl;
    cout << "[Time] final merge: ";
    cout << time[14] << " ms" << endl;
    cout << "[Time] multi scan: ";
    cout << time[11] << " ms" << endl;
    cout << "[Time] transfer result back to memory: ";
    cout << time[12] << " ms" << endl;
    // add new time counter here
    cout << "##############################" << endl;
    cout << endl;
  }

};
