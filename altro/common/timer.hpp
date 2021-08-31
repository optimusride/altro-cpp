// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <map>
#include <string>

namespace altro {

class Stopwatch;

/**
 * @brief Collects timing information from across many functions specified by
 * the user.
 * 
 * This class incorporates some overhead, so should only be used for timing 
 * functions that take greater than about 100 microseconds (overhead of creating
 * and destroying a stopwatch is about 10us on a desktop computer).
 * 
 * This class can only be instantiated as a pointer via either the MakeShared
 * or MakeUnique static methods.
 * 
 * NOTE: This timer is inactive by default. Use Activate() to gather profile 
 * info. If the timer is inactive, The creation of an empty Stopwatch object 
 * is very inexpensive and should have negligible impact on performance.
 * 
 * When a particular scope should be timed, a new Stopwatch type is generated
 * using Start. When the stopwatch goes out of scope it automatically 
 * logs the time with the timer. Each stopwatch can be provided a name to 
 * uniquely identify the code being profiled. Multiple calls to the same function
 * accumlate the overall time recorded by the timer.
 * 
 * A summary of the profiling results will be printed when the timer is 
 * destroyed, unless it has already been printed via PrintSummary().
 * 
 */
class Timer : public std::enable_shared_from_this<Timer> {
  using microseconds = std::chrono::microseconds; 
 public:
  ~Timer();
  static std::shared_ptr<Timer> MakeShared() {
    return std::shared_ptr<Timer>(new Timer());
  }
  static std::shared_ptr<Timer> MakeUnique() {
    return std::unique_ptr<Timer>(new Timer());
  }
  Stopwatch Start(const std::string& name);
  void PrintSummary();
  void PrintSummary(std::map<std::string, std::chrono::microseconds>* times);
  void Activate() { active_ = true; }
  void Deactivate() { active_ = false; }
  bool IsActive() const { return active_; }
  void SetOutput(FILE* io) { io_ = io; }  // ownership of resource remains with user
  void SetOutput(const std::string& filename);  // takes ownership of the file resource
  friend Stopwatch;  // so that it can populate times_

 private:
  Timer() = default;  // Make constructor private so it can only be created as a shared ptr
  std::vector<std::string> stack_;  // current profiler call stack e.g. "al/ilqr/cost"
  std::map<std::string, std::chrono::microseconds> times_;
  bool active_ = false;
  bool printed_summary_ = false;
  bool using_file_ = false;
  FILE* io_ = stdout;
};
using TimerPtr = std::shared_ptr<Timer>;

/**
 * @brief Collects and records the time taken from creation to destruction.
 * 
 * This class is used exclusively by the Timer class, which has exclusive 
 * access to it's constructor. Upon destruction, it automatically records the
 * time with the parent Timer that spanwed it.
 * 
 */
class Stopwatch {
  using microseconds = std::chrono::microseconds; 

 public:
  ~Stopwatch();
  friend Stopwatch Timer::Start(const std::string& name); // so that Timer can create a Stopwatch

 protected:
  Stopwatch() = default;
  Stopwatch(std::string name, std::shared_ptr<Timer> timer);

 private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::shared_ptr<Timer> parent_;
};

}  // namespace altro