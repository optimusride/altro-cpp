#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "examples/basic_constraints.hpp"
#include "examples/obstacle_constraints.hpp"

namespace altro {

constexpr int HEAP = Eigen::Dynamic;

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

TEST(CircleConstraint, Constructor) {
  examples::CircleConstraint obs;
  obs.AddObstacle(1.0, 2.0, 0.25);
  EXPECT_EQ(obs.OutputDimension(), 1);
  obs.AddObstacle(2.0, 4.0, 0.5);
  EXPECT_EQ(obs.OutputDimension(), 2);
}

TEST(CircleConstraint, Evaluate) {
  examples::CircleConstraint obs;
  Eigen::Vector2d p1(1.0, 2.0);
  Eigen::Vector2d p2(2.0, 4.0);
  obs.AddObstacle(p1(0), p1(1), 0.25);
  obs.AddObstacle(p2(0), p2(1), 0.5);

  const Eigen::Vector2d x(0.5, 1.5);
  const Eigen::Vector2d u(-0.25, 0.25);
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  obs.Evaluate(x, u, c);
  Eigen::Vector2d d1 = x - p1; 
  Eigen::Vector2d d2 = x - p2; 
  Eigen::Vector2d c_expected(0.25 * 0.25 - d1.squaredNorm(), 0.5 * 0.5 - d2.squaredNorm());
  EXPECT_TRUE(c.isApprox(c_expected));
}

TEST(CircleConstraint, Jacobian) {
  examples::CircleConstraint obs;
  Eigen::Vector2d p1(1.0, 2.0);
  Eigen::Vector2d p2(2.0, 4.0);
  obs.AddObstacle(p1(0), p1(1), 0.25);
  obs.AddObstacle(p2(0), p2(1), 0.5);

  const Eigen::Vector2d x(0.5, 1.5);
  const Eigen::Vector2d u(-0.25, 0.25);
  Eigen::Matrix2d jac = Eigen::Matrix2d::Zero();
  obs.Jacobian(x, u, jac);
  Eigen::Vector2d d1 = x - p1; 
  Eigen::Vector2d d2 = x - p2; 
  MatrixXd jac_expected(2,2);
  jac_expected << -2 * d1(0), -2 * d1(1), -2 * d2(0), -2 * d2(1);

  auto eval = [&](auto x_) {
    VectorXd c_(2);
    obs.Evaluate(x_, u, c_);
    return c_;
  };
  VectorXd x2 = x;
  MatrixXd jac_fd = utils::FiniteDiffJacobian<HEAP, HEAP>(eval, x2);

  EXPECT_TRUE(jac.isApprox(jac_expected));
  EXPECT_TRUE(jac.isApprox(jac_fd, 1e-4));
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
  Eigen::Vector4d x(4, 3, 2, 1);
  Eigen::Vector2d u(2, 3);

  // Evaluate method
  VectorXd c(n);
  VectorXd c2(n);
  Eigen::Vector4d c_expected(3, 1, -1, -3);
  goal->Evaluate(x, u, c);
  EXPECT_TRUE(c.isApprox(c_expected));
  conval.Evaluate(x, u, c2);
  EXPECT_TRUE(c2.isApprox(c_expected));

  // Jacobian method
  MatrixXd jac(n, n + m);
  MatrixXd jac2(n, n + m);
  MatrixXd jac_expected(n, n + m);
  jac_expected << MatrixXd::Identity(n, n), MatrixXd::Zero(n, m);
  goal->Jacobian(x, u, jac);
  EXPECT_TRUE(jac.isApprox(jac_expected));
  conval.Jacobian(x, u, jac2);
  EXPECT_TRUE(jac2.isApprox(jac_expected));
}

}  // namespace altro
