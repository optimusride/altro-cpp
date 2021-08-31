// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>
#include <chrono>

#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "examples/problems/triple_integrator.hpp"
#include "perf/benchmarks.hpp"

namespace altro {
namespace benchmarks {

template <int dof>
using ProbType = altro::problems::TripleIntegratorProblem<dof>;

void SolveTripleIntegrator(const bool add_constraints) {
  constexpr int dof = 2;
  problems::TripleIntegratorProblem<dof> prob_def;
  problem::Problem prob = prob_def.MakeProblem(add_constraints);

  constexpr int NStates = ProbType<dof>::NStates;
  constexpr int NControls = ProbType<dof>::NControls;
  augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> traj_ptr =
      std::make_shared<altro::Trajectory<NStates, NControls>>(prob_def.InitialTrajectory());
  solver.SetTrajectory(traj_ptr);

  if (add_constraints) {
    SetProfilerOptions(solver, "triple_integrator");
  } else {
    SetProfilerOptions(solver, "triple_integrator_uncon");
  }
  auto start = std::chrono::high_resolution_clock::now();
  // solver.GetOptions().verbose = LogLevel::kDebug;
  solver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration  = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  fmt::print("Total Compute Time: {:.4f} ms\n", duration.count());
}

}  // namespace benchmarks
}  // namespace altro

int main() {
  fmt::format("Unconstrained Triple Integrator:\n");
  bool add_constraints = false;
  altro::benchmarks::SolveTripleIntegrator(add_constraints);  

  fmt::format("Constrained Triple Integrator:\n");
  add_constraints = true;
  altro::benchmarks::SolveTripleIntegrator(add_constraints);  
  return 0;
}