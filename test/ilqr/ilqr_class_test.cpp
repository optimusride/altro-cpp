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

	// Must set the initial state and trajectory before solving
	iLQR<HEAP, HEAP> ilqr2(N);
	ilqr2.CopyFromProblem(prob, 0, N + 1);
  if (utils::AssertionsActive()) {
    EXPECT_DEATH(ilqr2.Solve(), "Assert.*Initial state must be set");
    ilqr2.SetInitialState(VectorXd::Zero(prob.GetDynamics(0)->StateDimension()));
    EXPECT_DEATH(ilqr2.Solve(), "Assert.*Invalid trajectory pointer");
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
  solver.Initialize();
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