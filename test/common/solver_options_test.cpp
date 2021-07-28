#include <fmt/format.h>
#include <fmt/color.h>
#include <gtest/gtest.h>

#include "altro/common/solver_stats.hpp"

namespace altro {

TEST(SolverStatsTest, LogCounts) {
  fmt::print("Hello there!\n");
  SolverStats stats;
  stats.SetVerbosity(LogLevel::kInnerDebug);
  EXPECT_EQ(stats.cost.size(), 0);
  EXPECT_EQ(stats.violations.size(), 0);

  // Should automatically create the first iteration
  stats.Log("cost", 10.0);
  EXPECT_EQ(stats.cost.size(), 1);
  EXPECT_EQ(stats.violations.size(), 1);  // all data fields should get larger

  // Size should be the same after logging the same field twice
  stats.Log("cost", 20.0);
  EXPECT_EQ(stats.cost.size(), 1);

  stats.Log("viol", 1e-2);
  stats.Log("viol", 1e-3);
  EXPECT_EQ(stats.violations.size(), 1);

  stats.NewIteration();
  stats.Log("cost", 2.0);
  EXPECT_EQ(stats.cost.size(), 2);
  EXPECT_EQ(stats.violations.size(), 2);

  EXPECT_DOUBLE_EQ(stats.cost[0], 20);
  EXPECT_DOUBLE_EQ(stats.cost[1], 2);
  EXPECT_DOUBLE_EQ(stats.violations[0], 1e-3);
  EXPECT_DOUBLE_EQ(stats.violations[1], 1e-3);  // should copy previous entry
}

void PseudoSolve(SolverStats& stats, LogLevel verbosity, int outer_iters = 5) {
  if (stats.GetVerbosity() < LogLevel::kInner) {
    stats.GetLogger().SetFrequency(10);
  }
  stats.Reset();
  stats.SetVerbosity(verbosity);
  stats.Log("iter_al", 0);
  stats.Log("viol", 10.1);
  stats.Log("pen", 0.85);
  for (int outer = 0; outer < outer_iters; ++outer) {
    stats.iterations_inner = 0;
    stats.initial_cost = 10.0;
    for (int inner = 0; inner < 10; ++inner) {
      // Backward pass
      stats.Log("reg", 10.0 - inner);
      // Forward pass
      stats.Log("cost", 5 - std::log((outer + 1) * 10 + inner));
      stats.Log("alpha", 0.1 + 0.1 * inner);
      stats.Log("z", 1.00 + (rand() % 100) / 1000.0);
      /// Update stats
      double dJ = 0.0;
      if (stats.iterations_inner == 0) {
        dJ = stats.initial_cost - stats.cost.back();
      } else {
        dJ = stats.cost.rbegin()[1] - stats.cost.rbegin()[0];
      }
      stats.iterations_inner++;
      stats.iterations_total++;
      stats.Log("dJ", dJ);
      stats.Log("viol", pow10(-outer) + std::exp2(-inner));
      stats.Log("grad", pow10(-outer) + pow10(-inner));
      stats.Log("iters", stats.iterations_total);
      stats.NewIteration();
      if (stats.GetVerbosity() >= LogLevel::kInner) {
        stats.PrintLast();
      }
    }

    // AL stats
    stats.iterations_outer++;
    stats.Log("pen", pow10(outer));
    stats.Log("iter_al", stats.iterations_outer);
    if (stats.GetVerbosity() < LogLevel::kInner) {
        stats.PrintLast();
    }
  }
}

TEST(SolverStatsTest, ChangeVerbosity) {
  SolverStats stats;
  fmt::print("\nVERBOSITY LEVEL {}\n", 4);
  PseudoSolve(stats, LogLevel::kInnerDebug);
  fmt::print("iters: total = {}, outer = {}, inner = {}\n", stats.iterations_total,
             stats.iterations_outer, stats.iterations_inner);

  stats.SetTolerances(1e-2, 1e-2, 1e-7);
  fmt::print("\nVERBOSITY LEVEL {}\n", 3);
  PseudoSolve(stats, LogLevel::kInner);
  fmt::print("iters: total = {}, outer = {}, inner = {}\n", stats.iterations_total,
             stats.iterations_outer, stats.iterations_inner);

  fmt::print("\nVERBOSITY LEVEL {}\n", 2);
  PseudoSolve(stats, LogLevel::kOuterDebug, 15);
  fmt::print("iters: total = {}, outer = {}, inner = {}\n", stats.iterations_total,
             stats.iterations_outer, stats.iterations_inner);

  fmt::print("\nVERBOSITY LEVEL {}\n", 1);
  PseudoSolve(stats, LogLevel::kOuter, 15);
  fmt::print("iters: total = {}, outer = {}, inner = {}\n", stats.iterations_total,
             stats.iterations_outer, stats.iterations_inner);

  fmt::print("\nVERBOSITY LEVEL {}\n", 0);
  PseudoSolve(stats, LogLevel::kSilent, 15);
  fmt::print("iters: total = {}, outer = {}, inner = {}\n", stats.iterations_total,
             stats.iterations_outer, stats.iterations_inner);
}

}  // namspace altro