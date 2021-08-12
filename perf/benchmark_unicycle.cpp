#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <chrono>

#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "examples/problems/unicycle.hpp"
#include "perf/benchmarks.hpp"

namespace altro {
namespace benchmarks {

void SolveUnicycle() {
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
  solver.GetOptions().nthreads = 4;

  auto start = std::chrono::high_resolution_clock::now();
  solver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::microseconds duration  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  fmt::print("Total Compute Time: {:.3f} ms\n", duration.count() / 1000.0);
}

}  // namespace benchmarks
}  // namespace altro

int main() {
  altro::benchmarks::SolveUnicycle();
  return 0;
}