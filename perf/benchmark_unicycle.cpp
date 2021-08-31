// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <chrono>
#include <iostream>

#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "examples/problems/unicycle.hpp"
#include "perf/benchmarks.hpp"

namespace altro {
namespace benchmarks {

void SolveUnicycle(int nthreads) {
  problems::UnicycleProblem prob_def;
  prob_def.SetScenario(problems::UnicycleProblem::kThreeObstacles);
  const bool add_constraints = true;
  problem::Problem prob = prob_def.MakeProblem(add_constraints);

  constexpr int NStates = problems::UnicycleProblem::NStates;
  constexpr int NControls = problems::UnicycleProblem::NControls;
  augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> traj_ptr =
      std::make_shared<altro::Trajectory<NStates, NControls>>(prob_def.InitialTrajectory());
  solver.SetTrajectory(traj_ptr);

  SetProfilerOptions(solver, "unicycle");
  solver.SetPenalty(10.0);
  solver.GetOptions().verbose = LogLevel::kDebug;
  solver.GetOptions().nthreads = nthreads;

  auto start = std::chrono::high_resolution_clock::now();
  solver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::microseconds duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  fmt::print("Total Compute Time: {:.3f} ms\n", duration.count() / 1000.0);
}

void SolveUnicycleLoop(int nruns, int nthreads) {
  problems::UnicycleProblem prob_def;
  prob_def.SetScenario(problems::UnicycleProblem::kThreeObstacles);
  const bool add_constraints = true;
  problem::Problem prob = prob_def.MakeProblem(add_constraints);

  constexpr int NStates = problems::UnicycleProblem::NStates;
  constexpr int NControls = problems::UnicycleProblem::NControls;
  augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> traj_ptr =
      std::make_shared<altro::Trajectory<NStates, NControls>>(
          prob_def.InitialTrajectory<NStates, NControls>());
  solver.SetTrajectory(traj_ptr);

  SetProfilerOptions(solver, "unicycle-loop");
  std::vector<std::chrono::duration<double, std::milli>> times;
  for (int iter = 0; iter < nruns; ++iter) {
    solver.SetPenalty(10.0);
    solver.GetOptions().verbose = LogLevel::kSilent;
    solver.GetOptions().nthreads = nthreads;
    *traj_ptr = prob_def.InitialTrajectory<NStates, NControls>();

    auto start = std::chrono::high_resolution_clock::now();
    solver.Solve();
    auto stop = std::chrono::high_resolution_clock::now();
    times.emplace_back(stop - start);
    fmt::print("Iteration {}: Cost = {}, iters = {}, Time = {}\n", iter,
               solver.GetiLQRSolver().Cost(), solver.GetStats().iterations_total, times.back());
    // fmt::print("  Total Compute Time: {:.3f} ms\n", times.back().count());
  }
}

}  // namespace benchmarks
}  // namespace altro

int main(int argc, char* argv[]) {
  int nruns = 1;
  if (argc > 1) {
    nruns = std::stoi(std::string(argv[1]));
  }

  int nthreads = 1;
  if (argc > 2) {
    nthreads = std::stoi(std::string(argv[2]));
  }

  if (nruns > 1) {
    altro::benchmarks::SolveUnicycleLoop(nruns, nthreads);
  } else {
    altro::benchmarks::SolveUnicycle(nthreads);
  }
  return 0;
}