#include <gtest/gtest.h>
#include <iostream>

#include "altro/ilqr/ilqr.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"
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
  // Originally should have N, but we created one more
  EXPECT_EQ(model.use_count(), N + 1);

  // Create a storage container for the base class
  std::vector<std::unique_ptr<ilqr::KnotPointFunctions<HEAP, HEAP>>> ptrs;

  // Add the first knot point
  ptrs.emplace_back(
      std::make_unique<ilqr::KnotPointFunctions<HEAP, HEAP>>(model, costfun));
  EXPECT_EQ(model.use_count(), N + 2);

  // Add the second knot point
  model = prob.GetDynamics(1);
  costfun = prob.GetCostFunction(1);
  ptrs.emplace_back(
      std::make_unique<ilqr::KnotPointFunctions<HEAP, HEAP>>(model, costfun));
  EXPECT_EQ(model.use_count(), N + 3);

  // Add a third knot point
  model = prob.GetDynamics(2);
  costfun = prob.GetCostFunction(2);
  ptrs.push_back(
      std::make_unique<ilqr::KnotPointFunctions<HEAP, HEAP>>(model, costfun));
  EXPECT_EQ(model.use_count(), N + 4);

  model = prob.GetDynamics(N);
  costfun = prob.GetCostFunction(N);
  EXPECT_EQ(model, nullptr);
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
  EXPECT_DEATH(ilqr.CopyFromProblem(prob_undefined, 0, N + 1),
               "Assert.*fully defined");

	// ilqr.CopyFromProblem(prob, N, N+1);
	EXPECT_DEATH(ilqr.CopyFromProblem(prob, N, N+1), "Assert.*copied in isolation");
}

}  // namespace ilqr
}  // namespace altro