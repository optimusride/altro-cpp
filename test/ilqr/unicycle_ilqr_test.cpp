// Copyright [2021] Optimus Ride Inc.

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>

#include "altro/augmented_lagrangian/al_problem.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/problem/problem.hpp"
#include "examples/basic_constraints.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/unicycle.hpp"
#include "test/test_utils.hpp"

class UnicycleiLQRTest : public altro::problems::UnicycleProblem, public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(UnicycleiLQRTest, BuildProblem) {
  altro::problem::Problem prob = MakeProblem();
  EXPECT_TRUE(prob.IsFullyDefined());
}

TEST_F(UnicycleiLQRTest, Initialization) {
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>();
  solver.Rollout();
  const double J_expected = 259.27636137767087;  // from Altro.jl
  EXPECT_LT(std::abs(solver.Cost() - J_expected), 1e-5);
}

TEST_F(UnicycleiLQRTest, BackwardPass) {
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>();
  solver.Rollout();
  solver.UpdateExpansions();
  solver.BackwardPass();

  Eigen::VectorXd ctg_grad0(n);
  Eigen::VectorXd d0(m);

  // These numbers were computed using Altro.jl
  ctg_grad0 << 0.024904637422419617, -0.46496022574032614, -0.0573096310550007;
  d0 << -2.565783457444465, 5.514158930898376;

  EXPECT_TRUE(solver.GetKnotPointFunction(0).GetCostToGoGradient().isApprox(ctg_grad0, 1e-5));
  EXPECT_TRUE(solver.GetKnotPointFunction(0).GetFeedforwardGain().isApprox(d0, 1e-5));
}

TEST_F(UnicycleiLQRTest, ForwardPass) {
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>();
  solver.Rollout();
  solver.UpdateExpansions();
  solver.BackwardPass();
  const double J0 = solver.Cost();
  solver.ForwardPass();
  EXPECT_LT(solver.Cost(), J0);
  EXPECT_DOUBLE_EQ(solver.GetStats().alpha[0], 0.0625);
}

TEST_F(UnicycleiLQRTest, TwoSteps) {
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>();
  solver.Rollout();
  solver.UpdateExpansions();
  solver.BackwardPass();
  solver.ForwardPass();

  solver.UpdateExpansions();
  solver.BackwardPass();

  // Values from Altro.jl
  Eigen::VectorXd ctg_grad0(n);
  Eigen::VectorXd d0(m);
  ctg_grad0 << -0.0015143873973949232, -0.07854630832127288, -0.017945283678268698;
  d0 << 0.21887571453613042, 1.3097976615154625;
  EXPECT_TRUE(solver.GetKnotPointFunction(0).GetCostToGoGradient().isApprox(ctg_grad0, 1e-5));
  EXPECT_TRUE(solver.GetKnotPointFunction(0).GetFeedforwardGain().isApprox(d0, 1e-5));

  solver.ForwardPass();
  const double J_expected = 62.773696055304384;  // from Altro.jl
  EXPECT_LT(solver.Cost() - J_expected, 1e-5);
}

TEST_F(UnicycleiLQRTest, FullSolve) {
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>();
  solver.GetOptions().verbose = altro::LogLevel::kInner;
  solver.Solve();
  const double J_expected = 0.0387016567;  // from Altro.jl
  const double iter_expected = 9;
  EXPECT_EQ(solver.GetStats().iterations_inner, iter_expected);
  EXPECT_EQ(solver.GetStatus(), altro::SolverStatus::kSolved);
  EXPECT_LT(std::abs(solver.Cost() - J_expected), 1e-5);
  EXPECT_LT(solver.GetStats().gradient.back(), solver.GetOptions().gradient_tolerance);
}

TEST_F(UnicycleiLQRTest, AugLagForwardPass) {
  bool alprob = true;
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>(alprob);
  solver.Rollout();
  solver.UpdateExpansions();
  solver.BackwardPass();
  const double J0 = solver.Cost();
  solver.ForwardPass();
  const double J = solver.Cost();
  EXPECT_LT(J, J0);
  EXPECT_DOUBLE_EQ(solver.GetStats().alpha[0], 0.0625);
}

TEST_F(UnicycleiLQRTest, AugLagFullSolve) {
  bool alprob = true;
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver<NStates, NControls>(alprob);
  solver.Solve();
  double J = solver.Cost();

  // Calculate the maximum violation
  std::shared_ptr<altro::Trajectory<NStates, NControls>> Z = solver.GetTrajectory();
  double v_max = 0;
  double w_max = 0;
  for (int k = 0; k < N; ++k) {
    double v = Z->Control(k)(0);
    double w = Z->Control(k)(1);
    v_max = std::max(std::abs(v), v_max);
    w_max = std::max(std::abs(w), w_max);
  }
  double max_violation = std::max(v_max - v_bnd, w_max - w_bnd);

  // from Altro.jl
  const double J_expected = 0.03893427133384412;  
  const double iter_expected = 10;
  const double max_violation_expected = 0.00017691645708972636; 

  // Compare to Altro.jl
  double cost_err = std::abs(J_expected - J) / J_expected;
  double viol_err = std::abs(max_violation_expected - max_violation) / max_violation_expected;
  EXPECT_LT(cost_err, 1e-6);
  EXPECT_LT(viol_err, 1e-6);
  EXPECT_EQ(solver.GetStats().iterations_inner, iter_expected);
}