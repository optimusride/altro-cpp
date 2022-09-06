//
// Created by brian on 9/6/22.
//

#include <chrono>
#include <fmt/core.h>
#include <fmt/chrono.h>

#include "examples/problems/unicycle.hpp"

void FullSolve() {
}

int main() {
  fmt::print("Hi there!\n");
  constexpr int NStates = altro::problems::UnicycleProblem::NStates;
  constexpr int NControls = altro::problems::UnicycleProblem::NControls;
  using millisf = std::chrono::duration<double, std::milli>;

  // Initialize problem
  altro::problems::UnicycleProblem unicycle_problem;
  altro::problem::Problem prob = unicycle_problem.MakeProblem();
  std::shared_ptr<altro::Trajectory<NStates, NControls>> Z =
      std::make_shared<altro::Trajectory<NStates, NControls>>(unicycle_problem.InitialTrajectory());

  // Initialize solver
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> alsolver(prob);
  alsolver.SetTrajectory(Z);
  alsolver.GetOptions().constraint_tolerance = 1e-6;
  alsolver.GetOptions().verbose = altro::LogLevel::kSilent;
  alsolver.GetiLQRSolver().GetOptions().verbose = altro::LogLevel::kInner;

  // Solve
  auto start = std::chrono::high_resolution_clock::now();
  alsolver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<millisf>(stop - start);
  fmt::print("Total time: {}\n", duration);

  // Print summary
  fmt::print("Total / Outer Iterations: {} / {}\n", alsolver.GetStats().iterations_total,
             alsolver.GetStats().iterations_outer);
  double J = alsolver.GetiLQRSolver().Cost();
  double viol = alsolver.GetMaxViolation();
  double pen = alsolver.GetMaxPenalty();
  fmt::print("Final cost / viol / penalty: {} / {} / {}\n", J, viol, pen);
  return EXIT_SUCCESS;
}