#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <cmath>
#include <chrono>

#include "altro/augmented_lagrangian/al_cost.hpp"
#include "altro/augmented_lagrangian/al_problem.hpp"
#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/assert.hpp"
#include "examples/basic_constraints.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/unicycle.hpp"
#include "test/test_utils.hpp"

namespace altro {
namespace augmented_lagrangian {

class AugLagTest : public problems::UnicycleProblem, public ::testing::Test {
 protected:
  const double v_violation = 0.5;
  const double rho = 1.1;

  Eigen::Vector3d x = Eigen::Vector3d(0.1, 0.2, M_PI / 3);
  Eigen::Vector2d u = Eigen::Vector2d(v_bnd + v_violation, w_bnd / 2);
  VectorNd<5> z;

  void SetUp() override { z << x, u; }

  template <int n2, int m2>
  ALCost<n2, m2> MakeALCost() {
    problem::Problem prob = MakeProblem();
    ALCost<n2, m2> alcost0(prob, 0);
    alcost0.GetInequalityConstraints().at(0)->SetPenalty(rho);
    alcost0.GetInequalityConstraints().at(0)->GetDuals()(2) = -rho * v_violation;
    return alcost0;
  }

  template <int n2, int m2>
  void TestALCostEval() {
    problem::Problem prob = MakeProblem();
    ALCost<n2, m2> alcost0(prob, 0);
    alcost0.GetInequalityConstraints().at(0)->SetPenalty(rho);

    double J_cost = qcost->Evaluate(x, u);
    double J_bnd = 0.5 * rho * v_violation * v_violation;
    double J_al = alcost0.Evaluate(x, u);
    EXPECT_DOUBLE_EQ(J_al, J_cost + J_bnd);

    // Set a non-zero dual variable
    alcost0.GetInequalityConstraints().at(0)->GetDuals()(2) = -rho * v_violation;
    J_al = alcost0.Evaluate(x, u);
    J_bnd = 0.5 * (std::pow(-2 * rho * v_violation, 2) - std::pow(rho * v_violation, 2)) / rho;
    EXPECT_DOUBLE_EQ(J_al, J_cost + J_bnd);
  }

  template <int n2, int m2>
  void TestALCostGradient() {
    ALCost<n2, m2> alcost0 = MakeALCost<n2, m2>();

    VectorNd<n2> dx(n);
    VectorNd<m2> du(m);
    alcost0.Gradient(x, u, dx, du);

    auto alcost = [&](auto z) { return alcost0.Evaluate(z.head(n), z.tail(m)); };
    VectorXd grad_fd = utils::FiniteDiffGradient(alcost, z);
    EXPECT_TRUE(dx.isApprox(grad_fd.head(n), 1e-4));
    EXPECT_TRUE(du.isApprox(grad_fd.tail(m), 1e-4));
  }

  template <int n2, int m2>
  void TestALCostHessian() {
    ALCost<n2, m2> alcost0 = MakeALCost<n2, m2>();
    MatrixNxMd<n2, n2> dxdx(n, n);
    MatrixNxMd<n2, m2> dxdu(n, m);
    MatrixNxMd<m2, m2> dudu(m, m);
    alcost0.Hessian(x, u, dxdx, dxdu, dudu);

    auto alcost = [&](auto z) { return alcost0.Evaluate(z.head(n), z.tail(m)); };
    MatrixXd hess_fd = utils::FiniteDiffHessian(alcost, z);
    EXPECT_TRUE(dxdx.isApprox(hess_fd.topLeftCorner(n, n), 1e-4));
    EXPECT_LT((dxdu - hess_fd.topRightCorner(n, m)).norm(), 1e-4);
    EXPECT_TRUE(dudu.isApprox(hess_fd.bottomRightCorner(m, m), 1e-4));
  }

  template <int n2, int m2>
  AugmentedLagrangianiLQR<n2, m2> MakeALSolver() {
    problem::Problem prob = MakeProblem();
    AugmentedLagrangianiLQR<n2, m2> alsolver(prob);
    std::shared_ptr<altro::Trajectory<n2, m2>> Z =
        std::make_shared<altro::Trajectory<NStates, NControls>>(
            InitialTrajectory<NStates, NControls>());
    alsolver.SetTrajectory(Z);
    return alsolver;
  }
};

TEST_F(AugLagTest, ALCostConstructor) {
  problem::Problem prob = MakeProblem();
  EXPECT_EQ(prob.NumSegments(), N);
  ALCost<NStates, NControls> alcost(n, m);
  EXPECT_EQ(alcost.StateDimension(), n);
  EXPECT_EQ(alcost.ControlDimension(), m);

  const int k = 0;  // knot point index
  ALCost<NStates, NControls> alcost0(prob, k);
  EXPECT_EQ(alcost0.NumConstraints(), prob.NumConstraints(k));
  EXPECT_EQ(alcost.NumConstraints(), 0);
}

TEST_F(AugLagTest, ALCostDeath) {
  problem::Problem prob(N);
  ALCost<NStates, NControls> alcost(n, m);
  const int k = 0;  // knot point index
  if (utils::AssertionsActive()) {
    EXPECT_DEATH(alcost.SetCostFunction(prob.GetCostFunction(k)), "Assert.*cannot be a nullptr");
  }
}

TEST_F(AugLagTest, ALCostEval) {
  TestALCostEval<NStates, NControls>();
  TestALCostEval<HEAP, HEAP>();
  TestALCostEval<NStates, HEAP>();
  TestALCostEval<HEAP, NControls>();
}

TEST_F(AugLagTest, ALCostGradient) {
  TestALCostGradient<NStates, NControls>();
  TestALCostGradient<HEAP, HEAP>();
  TestALCostGradient<NStates, HEAP>();
  TestALCostGradient<HEAP, NControls>();
}

TEST_F(AugLagTest, ALCostHessian) {
  TestALCostHessian<NStates, NControls>();
  TestALCostHessian<HEAP, HEAP>();
  TestALCostHessian<NStates, HEAP>();
  TestALCostHessian<HEAP, NControls>();
}

TEST_F(AugLagTest, SetALCostPenalty) {
  ALCost<NStates, NControls> alcost0 = MakeALCost<NStates, NControls>();
  std::vector<constraints::ConstraintPtr<constraints::Equality>> eq;
  eq.emplace_back(goal);
  alcost0.SetEqualityConstraints(eq.begin(), eq.end());

  // Set the penalties
  const double rho_goal = 11.0;
  const double rho_ubnd = 12.0;
  const double phi_goal = 2.5;
  const double phi_ubnd = 3.5;
  Eigen::VectorXd goal_penalty = Eigen::VectorXd::Constant(n, rho_goal);
  Eigen::VectorXd ubnd_penalty = Eigen::VectorXd::Constant(2 * m, rho_ubnd);

  // Get the constraint values
  const int con_idx = 0;  // constraint index
  std::shared_ptr<constraints::ConstraintValues<NStates, NControls, constraints::Equality>>
      goal_vals = alcost0.GetEqualityConstraints()[con_idx];
  std::shared_ptr<constraints::ConstraintValues<NStates, NControls, constraints::Inequality>>
      ubnd_vals = alcost0.GetInequalityConstraints()[con_idx];

  // Make sure the penalties update 
  alcost0.SetPenalty<constraints::Equality>(rho_goal, con_idx);
  alcost0.SetPenalty<constraints::Inequality>(rho_ubnd, con_idx);
  EXPECT_TRUE(goal_vals->GetPenalty().isApprox(goal_penalty));
  EXPECT_TRUE(ubnd_vals->GetPenalty().isApprox(ubnd_penalty));
  EXPECT_DOUBLE_EQ(goal_vals->MaxPenalty(), rho_goal);
  EXPECT_DOUBLE_EQ(ubnd_vals->MaxPenalty(), rho_ubnd);

  // Make sure the penalty scaling updates
  alcost0.SetPenaltyScaling<constraints::Equality>(phi_goal, con_idx);
  alcost0.SetPenaltyScaling<constraints::Inequality>(phi_ubnd, con_idx);
  EXPECT_DOUBLE_EQ(goal_vals->GetPenaltyScaling(), phi_goal);
  EXPECT_DOUBLE_EQ(ubnd_vals->GetPenaltyScaling(), phi_ubnd);

  // Run the dual update
  alcost0.UpdatePenalties();
  EXPECT_TRUE(goal_vals->GetPenalty().isApprox(phi_goal * goal_penalty));
  EXPECT_TRUE(ubnd_vals->GetPenalty().isApprox(phi_ubnd * ubnd_penalty));
  EXPECT_DOUBLE_EQ(goal_vals->MaxPenalty(), phi_goal * rho_goal);
  EXPECT_DOUBLE_EQ(ubnd_vals->MaxPenalty(), phi_ubnd * rho_ubnd);

  alcost0.SetPenalty<constraints::Equality>(2 * rho_goal);
  alcost0.SetPenalty<constraints::Inequality>(3 * rho_ubnd);
  EXPECT_DOUBLE_EQ(goal_vals->MaxPenalty(), 2 * rho_goal);
  EXPECT_DOUBLE_EQ(ubnd_vals->MaxPenalty(), 3 * rho_ubnd);
}

TEST_F(AugLagTest, CreateALProblem) {
  problem::Problem prob = MakeProblem();
  problem::Problem prob_al = BuildAugLagProblem<NStates, NControls>(prob);
  EXPECT_EQ(prob_al.NumSegments(), N);

  // Make sure the constraints are moved to the cost function
  EXPECT_GT(prob.NumConstraints(), 0);
  EXPECT_EQ(prob_al.NumConstraints(), 0);
  EXPECT_TRUE(prob.IsFullyDefined());
  EXPECT_TRUE(prob_al.IsFullyDefined());
}

TEST_F(AugLagTest, CreateiLQR) {
  problem::Problem prob = MakeProblem();
  problem::Problem prob_al = BuildAugLagProblem<NStates, NControls>(prob);
  ilqr::iLQR<NStates, NControls> solver(prob_al);
  EXPECT_EQ(solver.NumSegments(), N);
}

TEST_F(AugLagTest, ConstructSolver) {
  problem::Problem prob = MakeProblem();
  AugmentedLagrangianiLQR<NStates, NControls> alsolver(prob);
  EXPECT_EQ(alsolver.NumSegments(), N);
  EXPECT_EQ(alsolver.NumConstraints(), prob.NumConstraints());
}

TEST_F(AugLagTest, SolveiLQR) {
  problem::Problem prob = MakeProblem();
  AugmentedLagrangianiLQR<NStates, NControls> alsolver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> Z =
      std::make_shared<altro::Trajectory<NStates, NControls>>(
          InitialTrajectory<NStates, NControls>());
  alsolver.SetTrajectory(Z);

  // Solve first iLQR problem
  ilqr::iLQR<NStates, NControls>& ilqr_solver = alsolver.GetiLQRSolver();
  ilqr_solver.Solve();
  double J = ilqr_solver.Cost();
  double viol = alsolver.GetMaxViolation();

  // from Altro.jl
  const double J_expected = 0.03893427133384412;
  const double iter_expected = 10;
  const double max_violation_expected = 0.00017691645708972636;
  double cost_err = std::abs(J_expected - J) / J_expected;
  double viol_err = std::abs(max_violation_expected - viol) / max_violation_expected;
  EXPECT_LT(cost_err, 1e-6);
  EXPECT_LT(viol_err, 1e-6);
  EXPECT_EQ(ilqr_solver.GetStats().iterations_inner, iter_expected);
}

TEST_F(AugLagTest, TwoSolves) {
  problem::Problem prob = MakeProblem();
  AugmentedLagrangianiLQR<NStates, NControls> alsolver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> Z =
      std::make_shared<altro::Trajectory<NStates, NControls>>(
          InitialTrajectory<NStates, NControls>());
  alsolver.SetTrajectory(Z);
  std::shared_ptr<ALCost<NStates, NControls>> alcost_term = alsolver.GetALCost(N);
  std::shared_ptr<constraints::ConstraintValues<NStates, NControls, constraints::Equality>>
      goal_vals = alcost_term->GetEqualityConstraints()[0];

  // Solve first iLQR problem
  ilqr::iLQR<NStates, NControls>& ilqr_solver = alsolver.GetiLQRSolver();
  double J0 = ilqr_solver.Cost();
  double viol0 = alsolver.GetMaxViolation();
  ilqr_solver.Solve();
  double J = ilqr_solver.Cost();
  double viol = alsolver.GetMaxViolation();

  // Outer loop updates
  alsolver.UpdateDuals();
  alsolver.UpdatePenalties();
  fmt::print("Goal Duals:\n{}\n", goal_vals->GetDuals());
  double J_penalty = ilqr_solver.Cost();
  EXPECT_GT(J_penalty, J);
  EXPECT_LT(viol, viol0);
  EXPECT_LT(J, J0);

  // Run 2nd solve
  ilqr_solver.Solve();
  viol = alsolver.MaxViolation();
  const double viol_expected = 0.0000626;  // from Altro.jl
  const double iterations_expected = 1;
  double viol_err = std::abs(viol_expected - viol) / viol_expected;
  EXPECT_LT(viol_err, 0.1);
  EXPECT_EQ(ilqr_solver.GetStats().iterations_inner, iterations_expected);
  alsolver.UpdateDuals();
  alsolver.UpdatePenalties();

  fmt::print("Goal Duals:\n{}\n", goal_vals->GetDuals());

  // Run 3rd solve
  ilqr_solver.Solve();
  fmt::print("Iterations: {}\n", ilqr_solver.GetStats().iterations_inner);
  fmt::print("Cost: {}\n", ilqr_solver.Cost());
  fmt::print("Viol: {}\n", alsolver.GetMaxViolation());
  fmt::print("Penalty: {}\n", alsolver.GetMaxPenalty());
  fmt::print("alpha: {}\n", ilqr_solver.GetStats().alpha.back());
}

TEST_F(AugLagTest, FullSolve) {
  problem::Problem prob = MakeProblem();
  AugmentedLagrangianiLQR<NStates, NControls> alsolver(prob);
  std::shared_ptr<altro::Trajectory<NStates, NControls>> Z =
      std::make_shared<altro::Trajectory<NStates, NControls>>(
          InitialTrajectory<NStates, NControls>());
  alsolver.SetTrajectory(Z);

  alsolver.GetOptions().constraint_tolerance = 1e-6;
  alsolver.GetOptions().verbose = LogLevel::kSilent;
  alsolver.GetiLQRSolver().GetOptions().verbose = LogLevel::kInner;
  auto start = std::chrono::high_resolution_clock::now();
  alsolver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  fmt::print("Total time: {} ms\n", duration.count());
  fmt::print("Total / Outer Iterations: {} / {}\n", alsolver.GetStats().iterations_total,
             alsolver.GetStats().iterations_outer);
  double J = alsolver.GetiLQRSolver().Cost();
  double viol = alsolver.GetMaxViolation();
  double pen = alsolver.GetMaxPenalty();
  fmt::print("Final cost / viol / penalty: {} / {} / {}\n", J, viol, pen);
  EXPECT_EQ(alsolver.GetStatus(), SolverStatus::kSolved);
  EXPECT_LT(viol, alsolver.GetOptions().constraint_tolerance);
}

}  // namespace augmented_lagrangian
}  // namespace altro
