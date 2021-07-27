#include "altro/common/solver_stats.hpp"

#include <fmt/color.h>

namespace altro {

template <>
void SolverStats::SetPtr<double>(const LogEntry& entry, std::vector<double>& data) {
  floats_[entry.GetTitle()] = &data;
}

void SolverStats::SetTolerances(const double& cost, const double& viol, const double& grad) {
  logger_.GetEntry("dJ").SetLowerBound(cost);
  // logger_.GetEntry("dJ").SetUpperBound(cost);
  logger_.GetEntry("viol").SetLowerBound(viol);
  // logger_.GetEntry("viol").SetUpperBound(viol);
  logger_.GetEntry("grad").SetLowerBound(grad);
  // logger_.GetEntry("grad").SetUpperBound(grad);
}

void SolverStats::SetCapacity(int n) {
  for (auto kv : floats_) {
    kv.second->reserve(n);
  }
}

void SolverStats::Reset() {
  initial_cost = 0.0;
  iterations_inner = 0;
  iterations_total = 0;
  iterations_outer = 0;
  len_ = 0;
  int cur_capacity = cost.capacity();
  for (auto kv : floats_) {
    kv.second->clear();
  }
  logger_.Clear();
  SetCapacity(cur_capacity);
}

void SolverStats::NewIteration() {
  len_++;
  for (auto& kv : floats_) {
    kv.second->resize(len_);

    // Copy data from previous iteration
    if (len_ > 1) {
      kv.second->back() = kv.second->rbegin()[1];
    } else {
      kv.second->back() = 0.0;
    }
  }
}

void SolverStats::DefaultLogger() {
  logger_.AddEntry(0, "iters", "{:>4}", LogEntry::kInt)
      .SetName("iterations")
      .SetLevel(LogLevel::kOuterDebug)
      .SetWidth(6);  // NOLINT(readability-magic-numbers)
  logger_.AddEntry(1, "iter_al", "{:>4}", LogEntry::kInt)
      .SetName("iterations_outer")
      .SetLevel(LogLevel::kOuter)
      .SetWidth(8);  // NOLINT(readability-magic-numbers)
  SetPtr(logger_.AddEntry(-1, "cost", "{:>.4g}"), cost);
  SetPtr(logger_.AddEntry(-1, "viol", "{:>.3e}").SetName("constraint_violation"), violations);
  SetPtr(logger_.AddEntry(-1, "dJ", "{:>.2e}").SetName("cost_improvement"), cost_decrease);
  SetPtr(logger_.AddEntry(-1, "grad", "{:>.2e}").SetName("gradient").SetLevel(LogLevel::kOuterDebug), gradient);
  SetPtr(logger_.AddEntry(-1, "alpha", "{:>.2f}")
             .SetName("line_search_step_length")
             .SetLevel(LogLevel::kInner)
             .SetWidth(6),  // NOLINT(readability-magic-numbers)
         alpha);
  SetPtr(logger_.AddEntry(-1, "reg", "{:>.1e}")
             .SetName("regularization")
             .SetLevel(LogLevel::kInnerDebug)
             .SetWidth(7),  // NOLINT(readability-magic-numbers)
         regularization);
  SetPtr(logger_.AddEntry(-1, "z", "{:>.3f}")
             .SetName("cost_improvement_ratio")
             .SetLevel(LogLevel::kInnerDebug)
             .SetWidth(5),  // NOLINT(readability-magic-numbers)
         improvement_ratio);
  SetPtr(logger_.AddEntry(-1, "pen", "{:>.1e}")
             .SetName("max_penalty")
             .SetLevel(LogLevel::kDebug)
             .SetWidth(7),  // NOLINT(readability-magic-numbers)
         max_penalty);
  logger_.SetHeaderColor(fmt::color::yellow);
}

}  // namespace altro