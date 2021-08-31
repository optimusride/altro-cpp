// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <stdlib.h>

#include "altro/common/threadpool.hpp"
#include "perf/benchmarks.hpp"
#include "perf/task_launcher.hpp"

namespace altro {
namespace benchmarks {

void ThreadPoolTiming() {
  const int N = 16;
  const int nthreads = std::thread::hardware_concurrency() / 2;
  Launcher launcher(N, nthreads);
  ThreadPool pool;
  auto start = std::chrono::high_resolution_clock::now();
  launcher.WorkBlock(0, N);
  auto stop = std::chrono::high_resolution_clock::now();
  auto time_serial =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop - start);

  pool.LaunchThreads(nthreads);
  start = std::chrono::high_resolution_clock::now();
  launcher.CreateTasks(pool);
  pool.Wait();
  stop = std::chrono::high_resolution_clock::now();
  auto time_parallel =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop - start);
  double percent_expected = time_serial / time_parallel / nthreads * 100;
  std::string filename = "profile_threadpool.out";
  std::string outputfile = std::string(kLogDir) + "/" + filename;
  FILE* io = fopen(outputfile.c_str(), "w");
  fmt::print(io, "Number of threads: {}\n", nthreads);
  fmt::print(io, "Serial Time: {:0.2}\n", time_serial);
  fmt::print(io, "Parallel Time: {:0.2}\n", time_parallel);
  fmt::print(io, "Speedup: {:0.2f}x\n", time_serial / time_parallel);
  fmt::print(io, "Percent: {:0.2f}%\n", percent_expected);
  fclose(io);
}

}  // namespace benchmarks
}  // namespace altro

int main() {
  altro::benchmarks::ThreadPoolTiming();
}