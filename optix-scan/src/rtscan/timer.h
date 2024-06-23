// time count part
// semaphore

#ifndef TIME_H
#define TIME_H

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

  void showTime(int timeId, string title) {
    cout << "[Time] " << title << ": ";
    cout << time[timeId] << " ms" << endl;
  }
};

#endif
