#include <gtest/gtest.h>

#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/problem.hpp"
#include "examples/basic_constraints.hpp"

namespace altro {

TEST(BasicConstraints, ControlBoundConstructor) {
  int m = 3;
  double inf = std::numeric_limits<double>::infinity();
  examples::ControlBound bnd(m);
  EXPECT_EQ(bnd.OutputDimension(), 0);
  std::vector<double> lb = {-inf, -2, -3};
  bnd.SetLowerBound(lb);
  EXPECT_EQ(bnd.OutputDimension(), 2);

  lb = {-inf, 0, -inf};
  bnd.SetLowerBound(lb);
  EXPECT_EQ(bnd.OutputDimension(), 1);

  std::vector<double> ub = {inf, inf, inf};
  bnd.SetUpperBound(ub);
  EXPECT_EQ(bnd.OutputDimension(), 1);

  ub = {1, 2, 3};
  bnd.SetUpperBound(ub);
  EXPECT_EQ(bnd.OutputDimension(), 4);

	// Test moving bounds
	bnd.SetUpperBound(std::move(ub));
	bnd.SetLowerBound(std::move(lb));
	EXPECT_EQ(ub.size(), 0);
	EXPECT_EQ(lb.size(), 0);
}

TEST(BasicConstraints, GoalConstructor) {
  Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
  examples::GoalConstraint goal(xf);
  EXPECT_EQ(goal.OutputDimension(), 4);

  VectorXd xf2(xf);
  examples::GoalConstraint goal2(xf2);
  EXPECT_EQ(goal2.OutputDimension(), 4);
}


TEST(BasicConstraints, GoalConstraint) {
  Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
  examples::GoalConstraint goal(xf);
  VectorXd c(goal.OutputDimension());
  Eigen::Vector4d x(1, 2, 3, 4);
  Eigen::Vector3d u(-1, -2, -3);
  goal.Evaluate(x, u, c);
  EXPECT_TRUE(c.isApprox(Eigen::Vector4d::Zero()));
  goal.Evaluate(2 * x, u, c);
  EXPECT_TRUE(c.isApprox(x));
	VectorXd x_bad = VectorXd::Constant(5, 2.0);
	EXPECT_DEATH(goal.Evaluate(x_bad, u, c), "Assertion.*rows().*failed");
}


class ConstraintValueTest : public ::testing::Test {
 protected:
  static constexpr int n_static = 4;
  static constexpr int m_static = 2;
  int n;
  int m;
  constraints::ConstraintPtr<constraints::Equality> goal;
  constraints::ConstraintPtr<constraints::Inequality> ubnd;

  void SetUp() override {
    n = n_static;
    m = m_static;
    Eigen::Vector4d xf(1.0, 2.0, 3.0, 4.0);
    goal = std::make_shared<examples::GoalConstraint>(xf);

    std::vector<double> lb = {-2, -3};
    std::vector<double> ub = {2, 3};
    ubnd = std::make_shared<examples::ControlBound>(lb, ub);
  }
};

TEST_F(ConstraintValueTest, Constructor) {
  constraints::ConstraintValues<n_static, m_static, constraints::Equality> conval(n, m, goal);
  EXPECT_EQ(conval.StateDimension(), n);
  EXPECT_EQ(conval.ControlDimension(), m);
}

TEST_F(ConstraintValueTest, ConstraintInterface) {
  constraints::ConstraintValues<n_static, m_static, constraints::Equality> conval(n, m, goal);
  EXPECT_EQ(conval.OutputDimension(), goal->OutputDimension());

  // Some inputs
  Eigen::Vector4d x(4,3,2,1);
  Eigen::Vector2d u(2,3);

  // Evaluate method
  VectorXd c(n);
  VectorXd c2(n);
  Eigen::Vector4d c_expected(3,1,-1,-3);
  goal->Evaluate(x, u, c);
  EXPECT_TRUE(c.isApprox(c_expected));
  conval.Evaluate(x, u, c2);
  EXPECT_TRUE(c2.isApprox(c_expected));

  // Jacobian method
  MatrixXd jac(n, n+m);
  MatrixXd jac2(n, n+m);
  MatrixXd jac_expected(n, n+m);
  jac_expected << MatrixXd::Identity(n, n), MatrixXd::Zero(n, m);
  goal->Jacobian(x, u, jac);
  EXPECT_TRUE(jac.isApprox(jac_expected));
  conval.Jacobian(x, u, jac2);
  EXPECT_TRUE(jac2.isApprox(jac_expected));
}

}  // namespace altro
