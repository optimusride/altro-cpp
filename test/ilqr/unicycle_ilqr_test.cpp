#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "altro/common/trajectory.hpp"
#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/problem/problem.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/unicycle.hpp"

constexpr int n_static = 3;
constexpr int m_static = 2;
constexpr int HEAP = Eigen::Dynamic;

using ModelType = altro::problem::DiscretizedModel<altro::examples::Unicycle>;
using CostFunType = altro::examples::QuadraticCost;

class UnicycleiLQRTest : public ::testing::Test {
 protected:
  int N = 100;
  int n = n_static;
  int m = m_static;
  float tf = 3.0;
  float h = 0.03;
  Eigen::Matrix3d Q = Eigen::Vector3d::Constant(n_static, 1e-2).asDiagonal();
  Eigen::Matrix2d R = Eigen::Vector2d::Constant(m_static, 1e-2).asDiagonal();
  Eigen::Matrix3d Qf = Eigen::Vector3d::Constant(n_static, 100).asDiagonal();
  Eigen::Vector3d xf = Eigen::Vector3d(1.5, 1.5, M_PI / 2);
  Eigen::Vector3d x0 = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector2d u0 = Eigen::Vector2d::Constant(m_static, 0.1);

  void SetUp() override {
    Q *= h;
    R *= h;
  }

  altro::problem::Problem MakeProblem() {
    altro::problem::Problem prob(N);

    // Cost Function
    Eigen::Vector2d uref = Eigen::Vector2d::Zero();
    CostFunType qcost = CostFunType::LQRCost(Q, R, xf, uref);
    CostFunType qcost_term = CostFunType::LQRCost(Qf, R * 0, xf, uref, true);
    std::shared_ptr<CostFunType> costfun_ptr = std::make_shared<CostFunType>(qcost);
    std::shared_ptr<CostFunType> costfun_term_ptr = std::make_shared<CostFunType>(qcost_term);
    prob.SetCostFunction(costfun_ptr, 0, N);
    prob.SetCostFunction(costfun_term_ptr, N, N + 1);

    // Dynamics
    std::shared_ptr<ModelType> model = std::make_shared<ModelType>(altro::examples::Unicycle());
    prob.SetDynamics(model, 0, N);

    // Initial State
    prob.SetInitialState(x0);

    return prob;
  }

  template <int n_size, int m_size>
  altro::Trajectory<n_size, m_size> InitialTrajectory() {
    altro::Trajectory<n_size, m_size> Z(n, m, N);
    for (int k = 0; k < N; ++k) {
      Z.Control(k) = u0;
    }
    Z.SetUniformStep(h);
    return Z;
  }

  template <int n_size, int m_size>
  altro::ilqr::iLQR<n_size, m_size> MakeSolver() {
    altro::problem::Problem prob = MakeProblem();
    altro::ilqr::iLQR<n_size, m_size> solver(prob);

    std::shared_ptr<altro::Trajectory<n_size, m_size>> traj_ptr =
        std::make_shared<altro::Trajectory<n_size, m_size>>(InitialTrajectory<n_size, m_size>());

    solver.SetTrajectory(traj_ptr);
    return solver;
  }
};

TEST_F(UnicycleiLQRTest, BuildProblem) {
  altro::problem::Problem prob = MakeProblem();
  EXPECT_TRUE(prob.IsFullyDefined());
}

TEST_F(UnicycleiLQRTest, Initialization) {
  altro::ilqr::iLQR<n_static, m_static> solver = MakeSolver<n_static, m_static>();
  solver.Rollout();
  const double J_expected = 259.27636137767087;  // from Altro.jl
  EXPECT_LT(std::abs(solver.Cost() - J_expected), 1e-5);
}

TEST_F(UnicycleiLQRTest, BackwardPass) {
  altro::ilqr::iLQR<n_static, m_static> solver = MakeSolver<n_static, m_static>();
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
  altro::ilqr::iLQR<n_static, m_static> solver = MakeSolver<n_static, m_static>();
  solver.Rollout();
  solver.UpdateExpansions();
  solver.BackwardPass();
  const double J0 = solver.Cost();
  solver.ForwardPass();
  EXPECT_LT(solver.Cost(), J0);
  EXPECT_DOUBLE_EQ(solver.GetStats().alpha[0], 0.0625);
}

TEST_F(UnicycleiLQRTest, TwoSteps) {
  altro::ilqr::iLQR<n_static, m_static> solver = MakeSolver<n_static, m_static>();
  solver.Rollout();
  solver.UpdateExpansions();
  solver.BackwardPass();
  solver.ForwardPass();

  solver.UpdateExpansions();
  solver.BackwardPass();
  Eigen::VectorXd ctg_grad0(n);
  Eigen::VectorXd d0(m);
  ctg_grad0 << -0.0015143873973949232, -0.07854630832127288, -0.017945283678268698;
  d0 << 0.21887571453613042, 1.3097976615154625;
  EXPECT_TRUE(solver.GetKnotPointFunction(0).GetCostToGoGradient().isApprox(ctg_grad0, 1e-5));
  EXPECT_TRUE(solver.GetKnotPointFunction(0).GetFeedforwardGain().isApprox(d0, 1e-5));

  solver.ForwardPass();
  const double J_expected = 62.773696055304384;
  EXPECT_LT(solver.Cost() - J_expected, 1e-5);
}

TEST_F(UnicycleiLQRTest, FullSolve) {
  altro::ilqr::iLQR<n_static, m_static> solver = MakeSolver<n_static, m_static>();
  solver.Solve();
  const double J_expected = 0.0387016567;
  const double iter_expected = 9;
  EXPECT_EQ(solver.GetStats().iterations, iter_expected);
  EXPECT_EQ(solver.GetStatus(), altro::ilqr::SolverStatus::kSolved);
  EXPECT_LT(std::abs(solver.Cost() - J_expected), 1e-5);
  EXPECT_LT(solver.GetStats().gradient.back(), solver.GetOptions().gradient_tolerance);
}