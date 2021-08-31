// Copyright [2021] Optimus Ride Inc.

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <iostream>

#include "altro/ilqr/ilqr.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/augmented_lagrangian/al_solver.hpp"
#include "examples/problems/triple_integrator.hpp"

template <int dof>
using ProbType = altro::problems::TripleIntegratorProblem<dof>;

TEST(TripleIntegratorExample, Unconstrained) {
  // Create the problem
  ProbType<2> def;
  altro::problem::Problem prob = def.MakeProblem();

  // Create the solver
  constexpr int NStates = ProbType<2>::NStates;
  constexpr int NControls = ProbType<2>::NControls;
  altro::ilqr::iLQR<NStates, NControls> solver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> traj_ptr =
      std::make_shared<altro::Trajectory<NStates, NControls>>(def.InitialTrajectory());

  solver.SetTrajectory(traj_ptr);
  solver.Rollout();
  
  solver.Solve();
  EXPECT_EQ(solver.GetStatus(), altro::SolverStatus::kSolved);
  const int iterations_expected = 2;
  EXPECT_EQ(solver.GetStats().iterations_total, iterations_expected);
  EXPECT_LT(solver.GetStats().cost_decrease.back(), solver.GetOptions().cost_tolerance);
  EXPECT_LT(solver.GetStats().gradient.back(), solver.GetOptions().gradient_tolerance);
}

TEST(TripleIntegratorExample, Constrained) {
  // Create the problem
  constexpr int dof = 2;
  ProbType<dof> def;
  const bool add_constraints = true;
  altro::problem::Problem prob = def.MakeProblem(add_constraints);
  for (int k = 0; k < def.N; ++k) {
    EXPECT_EQ(prob.NumConstraints(k), 2 * dof);
  }
  EXPECT_EQ(prob.NumConstraints(def.N), 3 * dof);

  // Create the solver
  constexpr int NStates = ProbType<2>::NStates;
  constexpr int NControls = ProbType<2>::NControls;
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver(prob); 
  std::shared_ptr<altro::Trajectory<NStates, NControls>> traj_ptr =
      std::make_shared<altro::Trajectory<NStates, NControls>>(def.InitialTrajectory());

  solver.SetTrajectory(traj_ptr);
  
  solver.Solve();
  EXPECT_EQ(solver.GetStatus(), altro::SolverStatus::kSolved);
  EXPECT_LT(solver.GetStats().cost_decrease.back(), solver.GetOptions().cost_tolerance);
  EXPECT_LT(solver.GetStats().gradient.back(), solver.GetOptions().gradient_tolerance);
  EXPECT_LT(solver.GetStats().violations.back(), solver.GetOptions().constraint_tolerance);
  EXPECT_LT((traj_ptr->State(def.N) - def.xf).lpNorm<Eigen::Infinity>(), 
            solver.GetOptions().constraint_tolerance);

  Eigen::Vector2d ubnd(def.ubnd[0], def.ubnd[1]);
  EXPECT_TRUE(traj_ptr->Control(0).isApprox(ubnd));
  EXPECT_TRUE(traj_ptr->Control(def.N - 1).isApprox(ubnd));
}