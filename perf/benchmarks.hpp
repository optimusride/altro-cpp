// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>

#include "altro/augmented_lagrangian/al_solver.hpp"

namespace altro {
namespace benchmarks {

#ifndef LOCAL_LOG_DIR
constexpr const char* kLogDir = "";
#else
constexpr const char* kLogDir = LOCAL_LOG_DIR;
#endif

template <class Solver>
void SetProfilerOptions(Solver& solver, const std::string& basename) {
  solver.GetOptions().profiler_enable = true;
  solver.GetOptions().profiler_output_to_file = true;
  solver.GetOptions().log_directory = kLogDir;
  solver.GetOptions().profile_filename = fmt::format("profiler_{}.out", basename);
}

}  // namespace benchmarks
}  // namespace altro