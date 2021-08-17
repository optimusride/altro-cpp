#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <cmath>
#include <chrono>

#include "altro/common/timer.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/assert.hpp"
#include "examples/problems/unicycle.hpp"
#include "test/test_utils.hpp"

namespace altro {
namespace ilqr {

constexpr int HEAP = Eigen::Dynamic;

TEST(iLQRClassTest, Construction) {
  problem::Problem prob = MakeProblem();
  int N = prob.NumSegments();
  iLQR<HEAP, HEAP> ilqr(N);
  EXPECT_EQ(ilqr.NumSegments(), N);
}

TEST(iLQRClassTest, ReferenceCounts) {
  problem::Problem prob = MakeProblem();
  int N = prob.NumSegments();
  iLQR<HEAP, HEAP> ilqr(N);

  EXPECT_TRUE(prob.IsFullyDefined());

  // Get Pointers to model and cost function objects
  std::shared_ptr<problem::DiscreteDynamics> model = prob.GetDynamics(0);
  std::shared_ptr<problem::CostFunction> costfun = prob.GetCostFunction(0);
  // Originally should have 1, but we created one more
  // There's only 1 since it should be copied when creating the problem
  EXPECT_EQ(model.use_count(), 2);

  // Create a storage container for the base class
  std::vector<std::unique_ptr<ilqr::KnotPointFunctions<HEAP, HEAP>>> ptrs;

  // Add the first knot point
  ptrs.emplace_back(
      std::make_unique<ilqr::KnotPointFunctions<HEAP, HEAP>>(model, costfun));
  EXPECT_EQ(model.use_count(), 3);

  // Add the second knot point
  model = prob.GetDynamics(1);
  costfun = prob.GetCostFunction(1);
  ptrs.emplace_back(
      std::make_unique<ilqr::KnotPointFunctions<HEAP, HEAP>>(model, costfun));
  EXPECT_EQ(model.use_count(), 3);

  // Add a third knot point
  model = prob.GetDynamics(2);
  costfun = prob.GetCostFunction(2);
  ptrs.push_back(
      std::make_unique<ilqr::KnotPointFunctions<HEAP, HEAP>>(model, costfun));
  EXPECT_EQ(model.use_count(), 3);

  model = prob.GetDynamics(N);
  costfun = prob.GetCostFunction(N);
  EXPECT_EQ(model->StateDimension(), prob.GetDynamics(N-1)->StateDimension());
}

TEST(iLQRClassTest, CopyFromProblem) {
  problem::Problem prob = MakeProblem();
  int N = prob.NumSegments();
  iLQR<HEAP,HEAP> ilqr(N);
  ilqr.CopyFromProblem(prob, 0, N + 1);
  EXPECT_EQ(ilqr.NumSegments(), N);

  iLQR<6,2> ilqr2(N);
  ilqr2.InitializeFromProblem(prob);
  EXPECT_EQ(ilqr2.NumSegments(), N);
}

TEST(iLQRClassTest, InitialCondition) {
  problem::Problem prob = MakeProblem();
  iLQR<6,2> ilqr(prob.NumSegments());
  VectorXd x0(6);
  x0 << 1,2,3,4,5,6;
  prob.SetInitialState(x0);
  ilqr.InitializeFromProblem(prob);
  EXPECT_TRUE(ilqr.GetInitialState()->isApprox(x0));
  
  // Check that calling SetInitialState on the problem changes the solver
  prob.SetInitialState(2 * x0);
  EXPECT_TRUE(ilqr.GetInitialState()->isApprox(2 * x0));
}

TEST(iLQRClassTest, DeathTests) {
  problem::Problem prob = MakeProblem();
  int N = prob.NumSegments();
  iLQR<HEAP,HEAP> ilqr(N);

  problem::Problem prob_undefined(N);
  if (utils::AssertionsActive()) {
    EXPECT_DEATH(ilqr.CopyFromProblem(prob_undefined, 0, N + 1),
                 "Assert.*fully defined");
  }

  if (utils::AssertionsActive()) {
    // Must set the initial state and trajectory before solving
    iLQR<HEAP, HEAP> ilqr2(N);
    ilqr2.CopyFromProblem(prob, 0, N + 1);
    EXPECT_DEATH(ilqr2.Solve(), "Assert.*Invalid trajectory pointer");

    // Check dimensions
    iLQR<7,2> ilqr3(N);
    EXPECT_DEATH(ilqr3.InitializeFromProblem(prob), "Assert.*Inconsistent state dimension");
    iLQR<6,3> ilqr4(N);
    EXPECT_DEATH(ilqr4.InitializeFromProblem(prob), "Assert.*Inconsistent control dimension");

    // Copy a problem of different length
    problem::Problem prob2 = MakeProblem(2, 20);
    iLQR<6,2> ilqr5(20);
    ilqr5.InitializeFromProblem(prob2);
    EXPECT_EQ(ilqr5.NumSegments(), 20);
    EXPECT_DEATH(ilqr5.InitializeFromProblem(prob), "Assert.*Number of segments");
  }
}

TEST(iLQRClassTest, ParallelExpansion) {
  problems::UnicycleProblem def;
  def.N = 100;
  const int Nx = def.NStates;
  const int Nu = def.NControls;
  iLQR<Nx, Nu> solver = def.MakeSolver();

  const int Nruns = 2;
  {
    for (int i = 0; i < Nruns; ++i) {
      solver.UpdateExpansions();
    }
  }
  std::vector<MatrixXd> jacs;
  for (int k = 0; k < solver.NumSegments(); ++k) {
    jacs.emplace_back(solver.GetKnotPointFunction(k).GetDynamicsExpansion().GetJacobian());
  }

  solver.GetStats().GetTimer()->Activate();
  solver.GetOptions().nthreads = altro::kPickHardwareThreads; 
  solver.SolveSetup();
  {
    for (int i = 0; i < Nruns; ++i) {
      solver.UpdateExpansions();
    }
  }

  for (int k = 0; k < solver.NumSegments(); ++k) {
    EXPECT_TRUE(jacs[k].isApprox(solver.GetKnotPointFunction(k).GetDynamicsExpansion().GetJacobian()));
  }
}

}  // namespace ilqr
}  // namespace altro