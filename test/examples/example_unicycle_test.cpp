#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <iostream>

#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/common/solver_options.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "examples/problems/unicycle.hpp"

class UnicycleExampleTest : public altro::problems::UnicycleProblem, public ::testing::Test {
 protected:
  void SetUp() override { SetScenario(kThreeObstacles); }
};

TEST_F(UnicycleExampleTest, Construction) {
  altro::problem::Problem prob = MakeProblem(true);
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver();
  double J = solver.Cost();
  const double J_expected = 133.1151550141444;  // from Altro.jl
  EXPECT_LT(std::abs(J - J_expected), 1e-6);

  const bool al_cost = true;
  altro::ilqr::iLQR<NStates, NControls> solver_al = MakeSolver(al_cost);
  J = solver_al.Cost();
  const double Jal_expected = 141.9639680271223;
  EXPECT_LT(std::abs(J - Jal_expected), 1e-6);
}

TEST_F(UnicycleExampleTest, IncreasePenalty) {
  altro::problem::Problem prob = MakeProblem(true);

  // Create iLQR solver w/ AL objective
  const bool al_cost = true;
  altro::ilqr::iLQR<NStates, NControls> solver = MakeSolver(al_cost);
  double J0 = solver.Cost();

  // Create AL-iLQR Solver
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al(prob);
  solver_al.SetTrajectory(std::make_shared<altro::Trajectory<NStates, NControls>>(InitialTrajectory()));
  solver_al.GetiLQRSolver().Rollout();
  double J = solver_al.GetiLQRSolver().Cost();

  EXPECT_DOUBLE_EQ(J0, J);

  solver_al.SetPenalty(10.0);
  double J_expected = 221.6032851439234;  // from Altro.jl
  J = solver_al.GetiLQRSolver().Cost();
  EXPECT_LT(std::abs(J_expected - J), 1e-6);
}

TEST_F(UnicycleExampleTest, SolveOneStep) {
  altro::problem::Problem prob = MakeProblem(true);
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al(prob);
  solver_al.SetTrajectory(std::make_shared<altro::Trajectory<NStates, NControls>>(InitialTrajectory()));
  solver_al.GetiLQRSolver().Rollout();
  solver_al.SetPenalty(10.0);

  altro::ilqr::iLQR<NStates, NControls>& ilqr = solver_al.GetiLQRSolver();
  ilqr.Solve();
  solver_al.UpdateDuals();
  solver_al.UpdatePenalties();

  Eigen::VectorXd lambdaN(n);
  lambdaN << 0.43555910438329626, -0.5998598475208317, 0.0044282251970790935; // from Altro.jl
  EXPECT_TRUE(solver_al.GetALCost(N)->GetEqualityConstraints()[0]->GetDuals().isApprox(-lambdaN, 1e-6));
}

TEST_F(UnicycleExampleTest, SolveConstrained) {
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al = MakeALSolver();
  solver_al.SetPenalty(10.0);
  solver_al.GetOptions().verbose = altro::LogLevel::kDebug;
  solver_al.Solve();

  int num_obstacles = cx.size();
  for (int k = 0; k <= N; ++k) {
    double px = solver_al.GetiLQRSolver().GetTrajectory()->State(k)[0];
    double py = solver_al.GetiLQRSolver().GetTrajectory()->State(k)[1];
    for (int i = 0; i < num_obstacles; ++i) {
      altro::examples::Circle obs(cx(i), cy(i), cr(i));
      EXPECT_GT(obs.Distance(px, py), -1e-3); // 1 mm
    }
  }

  EXPECT_EQ(solver_al.GetStatus(), altro::SolverStatus::kSolved);
  EXPECT_LT(solver_al.MaxViolation(), solver_al.GetOptions().constraint_tolerance);
  EXPECT_LT(solver_al.GetStats().cost_decrease.back(), solver_al.GetOptions().cost_tolerance);
  EXPECT_LT(solver_al.GetStats().gradient.back(), solver_al.GetOptions().gradient_tolerance);
}