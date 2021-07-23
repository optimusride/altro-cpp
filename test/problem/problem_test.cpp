#include <gtest/gtest.h>
#include <iostream>

#include "altro/constraints/constraint.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/problem/dynamics.hpp"
#include "altro/problem/problem.hpp"
#include "examples/basic_constraints.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/triple_integrator.hpp"
#include "test/test_utils.hpp"

namespace altro {
namespace problem {

TEST(ProblemTests, Initialization) {
  int N = 10;
  Problem prob(N);
  EXPECT_EQ(prob.NumSegments(), N);
  EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, AddDynamics) {
  int N = 10;
  Problem prob(N);
  ModelPtr model_ptr = MakeModel();
  prob.SetDynamics(model_ptr, 0);
  EXPECT_FALSE(prob.IsFullyDefined());
  EXPECT_NE(prob.GetDynamics(0), nullptr);
  EXPECT_DEATH(prob.GetDynamics(1), "Assert.*Dynamics have not been defined.");
  prob.SetDynamics(model_ptr, 0, N);
  for (int k = 0; k < N; ++k) {
    EXPECT_NE(prob.GetDynamics(k), nullptr);
  }
  EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, AddCosts) {
  int N = 10;
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
  int N = 10;
  Problem prob(N);
  VectorXd x0 = VectorXd::Random(6);
  prob.SetInitialState(x0);
  EXPECT_TRUE(prob.GetInitialState().isApprox(x0));
}

TEST(ProblemTests, ChangeInitialState) {
  int N = 10;
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
  prob.SetCostFunction(costfun_ptr, 0, N + 1);
  prob.SetInitialState(x0);

  EXPECT_TRUE(prob.IsFullyDefined());

  VectorXd x0_bad = VectorXd::Random(7);
  prob.SetInitialState(x0_bad);

  EXPECT_FALSE(prob.IsFullyDefined());
}

TEST(ProblemTests, AddConstraints) {
  int N = 10;
  problem::Problem prob(N);

  // Goal Constraint
  Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
  constraints::ConstraintPtr<constraints::Equality> goal =
      std::make_shared<examples::GoalConstraint>(xf);
  prob.SetConstraint(goal, N);
  EXPECT_EQ(prob.NumConstraints(N), 4);

  // Control Bound Constraint
  std::vector<double> lb = {-2, -3};
  std::vector<double> ub = {2, 3};
  constraints::ConstraintPtr<constraints::Inequality> ubnd =
      std::make_shared<examples::ControlBound>(lb, ub);
  EXPECT_EQ(prob.NumConstraints(1), 0);
  EXPECT_EQ(ubnd->OutputDimension(), 4);
  for (int k = 0; k < N; ++k) {
    prob.SetConstraint(ubnd, k);
  }
  EXPECT_EQ(prob.NumConstraints(0), 4);
  EXPECT_EQ(prob.NumConstraints(N - 1), 4);
}

TEST(ProblemTests, AddConstraintsDeath) {
  int N = 10;
  int m = 2;
  problem::Problem prob(N);

  // Goal Constraint
  constraints::ConstraintPtr<constraints::Equality> goal;
  EXPECT_DEATH(prob.SetConstraint(goal, N), "Assert.*provide a valid constraint pointer");

  // Control Bound Constraint
  constraints::ConstraintPtr<constraints::Inequality> ubnd =
      std::make_shared<examples::ControlBound>(m);
  EXPECT_DEATH(prob.SetConstraint(ubnd, 0), "Assert.*length greater than zero");

  constraints::ConstraintPtr<constraints::Inequality> ubnd2;
  EXPECT_DEATH(prob.SetConstraint(ubnd2, 0), "Assert.*provide a valid constraint pointer");
}

}  // namespace problem
}  // namespace altro
