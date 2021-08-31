// Copyright [2021] Optimus Ride Inc.

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

TEST_F(UnicycleExampleTest, SolveParallel) {
  N = 10;
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NStates, NControls> solver_al = MakeALSolver();
  // solver_al.SetPenalty(10.0);
  solver_al.GetOptions().initial_penalty = 10.0;
  solver_al.GetOptions().verbose = altro::LogLevel::kDebug;

  altro::Trajectory<NStates, NControls> Z0 = InitialTrajectory<NStates, NControls>();
  solver_al.Init();


  // Compare expansions after 2 iLQR solves
  altro::ilqr::iLQR<NStates, NControls>& ilqr = solver_al.GetiLQRSolver();
  *(ilqr.GetTrajectory()) = Z0;
  auto pen = solver_al.GetALCost(0)->GetInequalityConstraints()[0]->GetPenalty();
  EXPECT_TRUE(pen.isApproxToConstant(10.0));
  ilqr.SolveSetup();
  ilqr.Rollout();

  auto step = [&ilqr](int iter) {
    for (int i = 0; i < iter; ++i) {
      ilqr.UpdateExpansions();
      ilqr.BackwardPass();
      ilqr.ForwardPass();
      ilqr.UpdateConvergenceStatistics();
    }
  };

  step(2);
  std::vector<Eigen::MatrixXd> jacs;
  std::vector<Eigen::MatrixXd> Qxx;
  std::vector<Eigen::MatrixXd> Quu;
  std::vector<Eigen::MatrixXd> K;
  for (int k = 0; k < ilqr.NumSegments(); ++k) {
    jacs.emplace_back(ilqr.GetKnotPointFunction(k).GetDynamicsExpansion().GetJacobian());
    Qxx.emplace_back(ilqr.GetKnotPointFunction(k).GetCostExpansion().dxdx());
    Quu.emplace_back(ilqr.GetKnotPointFunction(k).GetCostExpansion().dudu());
    K.emplace_back(ilqr.GetKnotPointFunction(k).GetFeedbackGain());
  }

  *(ilqr.GetTrajectory()) = Z0;
  ilqr.GetStats().Reset();
  ilqr.GetOptions().nthreads = 2;

  ilqr.SolveSetup();
  ilqr.Rollout();
  step(2);
  for (int k = 0; k < ilqr.NumSegments(); ++k) {
    Eigen::MatrixXd jac = ilqr.GetKnotPointFunction(k).GetDynamicsExpansion().GetJacobian();
    bool dynamics = jacs[k].isApprox(jac);
    bool qxx = Qxx[k].isApprox(ilqr.GetKnotPointFunction(k).GetCostExpansion().dxdx());
    bool quu = Quu[k].isApprox(ilqr.GetKnotPointFunction(k).GetCostExpansion().dudu());
    bool gain = K[k].isApprox(ilqr.GetKnotPointFunction(k).GetFeedbackGain());
    // fmt::print("Index {}: Dynamics? {}, Qxx? {}, Quu? {}, Gain? {}\n", k, dynamics, qxx, quu, gain);
    // if (!dynamics) {
    //   fmt::print("Expected:\n{}\n", jacs[k]);
    //   fmt::print("Got:\n{}\n", jac);
    // }
    EXPECT_TRUE(dynamics);
    EXPECT_TRUE(qxx);
    EXPECT_TRUE(quu);
    EXPECT_TRUE(gain);
  }

  // Compare entire solves
  *(ilqr.GetTrajectory()) = Z0;
  solver_al.GetOptions().nthreads = 1;
  solver_al.Solve();
  double cost = ilqr.Cost();
  int iters = solver_al.GetStats().iterations_total;

  *(ilqr.GetTrajectory()) = Z0;
  solver_al.GetOptions().nthreads = 2;
  solver_al.Solve();
  EXPECT_DOUBLE_EQ(cost, ilqr.Cost());
  EXPECT_EQ(iters, ilqr.GetStats().iterations_total);
}