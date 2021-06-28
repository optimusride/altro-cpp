#include <gtest/gtest.h>
#include <iostream>

#include "test/test_utils.hpp"
#include "altro/problem/problem.hpp"
#include "altro/problem/dynamics.hpp"
#include "altro/problem/discretized_model.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/triple_integrator.hpp"

namespace altro {
namespace problem {


TEST(ProblemTests, Initialization) {
	int N = 11;
	Problem prob(N);
	EXPECT_EQ(prob.NumSegments(), N);
	EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, AddDynamics) {
	int N = 11;
	Problem prob(N);
	ModelPtr model_ptr = MakeModel(); 
	prob.SetDynamics(model_ptr, 0);
	EXPECT_FALSE(prob.IsFullyDefined());
	EXPECT_NE(prob.GetDynamics(0), nullptr);
	EXPECT_EQ(prob.GetDynamics(1), nullptr);
	prob.SetDynamics(model_ptr, 0, N);
	for (int k = 0; k < N; ++k) {
		EXPECT_NE(prob.GetDynamics(k), nullptr);
	}
	EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, AddCosts) {
	int N = 11;
	Problem prob(N);
	CostPtr costfun_ptr = MakeCost();
	prob.SetCostFunction(costfun_ptr, 5);
	EXPECT_NE(prob.GetCostFunction(5), nullptr);
	EXPECT_EQ(prob.GetCostFunction(0), nullptr);

	prob.SetCostFunction(costfun_ptr, 0, 4);
	EXPECT_NE(prob.GetCostFunction(0), nullptr);
	EXPECT_NE(prob.GetCostFunction(1), nullptr);
	EXPECT_NE(prob.GetCostFunction(2), nullptr);
	EXPECT_NE(prob.GetCostFunction(3), nullptr);
	EXPECT_EQ(prob.GetCostFunction(4), nullptr);
	EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, DynamicsAndCosts) {
	int N = 10;
	Problem prob(N);
	CostPtr costfun_ptr = MakeCost();
	ModelPtr model_ptr = MakeModel(); 

	prob.SetDynamics(model_ptr, 0, N);
	prob.SetCostFunction(costfun_ptr, 0, N);
	EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, InitialState) {
	int N = 11;
	Problem prob(N);
	VectorXd x0 = VectorXd::Random(6);
	prob.SetInitialState(x0);
	EXPECT_TRUE(prob.GetInitialState().isApprox(x0));
}

TEST(ProblemTests, ChangeInitialState) {
	int N = 11;
	Problem prob(N);
	VectorXd x0 = VectorXd::Random(6);
	prob.SetInitialState(x0);
	EXPECT_TRUE(prob.GetInitialState().isApprox(x0));
	VectorXd x0_modified = VectorXd::Random(6);
	prob.SetInitialState(x0_modified);
	EXPECT_TRUE(prob.GetInitialState().isApprox(x0_modified));
}

TEST(ProblemTests, FullyDefined) {
	int N = 10;
	Problem prob(N);
	CostPtr costfun_ptr = MakeCost();
	ModelPtr model_ptr = MakeModel(); 
	VectorXd x0 = VectorXd::Random(6);

	prob.SetDynamics(model_ptr, 0, N);
	prob.SetCostFunction(costfun_ptr, 0, N+1);
	prob.SetInitialState(x0);

	EXPECT_TRUE(prob.IsFullyDefined());

	VectorXd x0_bad = VectorXd::Random(7);
	prob.SetInitialState(x0_bad);

	EXPECT_FALSE(prob.IsFullyDefined());
}

}  // namespace problem
}  // namespace altro
