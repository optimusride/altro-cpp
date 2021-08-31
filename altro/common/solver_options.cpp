// Copyright [2021] Optimus Ride Inc.

#include "altro/common/solver_options.hpp"

namespace altro {

#ifndef LOGDIR
constexpr const char* kLogDirectory = "logs";
#else
constexpr const char* kLogDirectory = LOGDIR;
#endif

SolverOptions::SolverOptions() : log_directory(kLogDirectory) {}
}  // namespace altro