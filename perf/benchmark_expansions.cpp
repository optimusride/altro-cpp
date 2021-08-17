#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

#include "altro/common/timer.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"
#include "examples/problems/unicycle.hpp"
#include "perf/benchmarks.hpp"

namespace altro {
namespace benchmarks {

template <int Nx, int Nu, class Time>
void RunTest(ilqr::iLQR<Nx, Nu>& solver, Time tserial, int Nruns, int nthreads,
             int tasks_per_thread = 1, FILE* io = stdout) {
  solver.GetOptions().nthreads = nthreads;
  solver.GetOptions().tasks_per_thread = tasks_per_thread;
  solver.SolveSetup();
  std::chrono::duration<double, std::micro> tparallel;
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Nruns; ++i) {
      solver.UpdateExpansions();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    tparallel = stop - start;
  }
  fmt::print(io, "Parallel w/ {} tasks and {} threads: {:.2f}x speedup, {:.2f}% expected\n",
             solver.NumTasks(), solver.NumThreads(), tserial / tparallel,
             tserial / tparallel / solver.NumThreads() * 100);
}

void ParallelExpansions() {
  problems::UnicycleProblem def;
  def.N = 100;
  const int Nx = def.NStates;
  const int Nu = def.NControls;
  ilqr::iLQR<Nx, Nu> solver = def.MakeSolver();
  int nprocs = std::thread::hardware_concurrency();
  std::string filename = "profile_expansions.out";
  std::string filepath = std::string(kLogDir) + "/" + filename;
  FILE* io = fopen(filepath.c_str(), "w");
  if (io == nullptr) {
    fmt::print("Couldn't write to file {}, writing to stdout instead\n", filepath);
    io = stdout;
  }
  fmt::print(io, "Number of threads: {}\n", nprocs);

  const int Nruns = 100;
  TimerPtr timer = Timer::MakeShared();
  solver.GetOptions().profile_filename = "profiler_expansions.out";
  solver.GetOptions().log_directory = LOCAL_LOG_DIR;
  timer->SetOutput(solver.GetStats().ProfileOutputFile());
  timer->Activate();

  // Serial
  std::chrono::duration<double, std::milli> tserial;
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Nruns; ++i) {
      solver.UpdateExpansions();
    }
    auto stop = std::chrono::high_resolution_clock::now();
    tserial = stop - start;
  }
  fmt::print(io, "Serial time: {}\n", tserial);

  // Serial w/ tasks
  solver.GetOptions().nthreads = altro::kPickHardwareThreads;
  std::vector<int> work_inds = solver.GetTaskAssignment();
  std::chrono::duration<double, std::micro> tserialtasks;
  int ntasks = solver.NumTasks();
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Nruns; ++i) {
      for (int j = 0; j < ntasks; ++j) {
        solver.UpdateExpansionsBlock(work_inds[j], work_inds[j + 1]);
      }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    tserialtasks = stop - start;
  }
  fmt::print(io, "Serial w/ {} tasks: {}x slower\n", ntasks, tserialtasks / tserial);

  RunTest(solver, tserial, Nruns, 4, 1, io);
  RunTest(solver, tserial, Nruns, 8, 1, io);
  RunTest(solver, tserial, Nruns, 8, 2, io);
  RunTest(solver, tserial, Nruns, 8, 4, io);
}

}  // namespace benchmarks
}  // namespace altro

int main() {
  altro::benchmarks::ParallelExpansions();
}