#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>

#include "altro/problem/dynamics.hpp"

namespace altro {
namespace problem {

template <int Nx, int Nu>
class TestDynamics : public ContinuousDynamics {
 public:
  using FunctionBase::Evaluate;
  using FunctionBase::Jacobian;
  using FunctionBase::Hessian;

  static constexpr int NStates = Nx;
  static constexpr int NControls = Nu;
  static constexpr int NOutputs = Nx;

  int StateDimension() const override { return 4; }
  int ControlDimension() const override { return 2; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                Eigen::Ref<VectorXd> out) override {
    out(0) = u(0) * t;
    out(1) = u(1) * t;
    out(2) = u(0) * x(0) + std::pow(x(2), 2);
    out(3) = u(1) * x(1) + std::pow(x(3), 2);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                JacobianRef jac) override {
    jac.setZero();
    jac(0, 4) = t;
    jac(1, 5) = t;
    jac(2, 0) = u(0);
    jac(2, 2) = 2 * x(2);
    jac(2, 4) = x(0);
    jac(3, 1) = u(1);
    jac(3, 3) = 2 * x(3);
    jac(3, 5) = x(1);
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    hess(0, 4) = b(2);
    hess(1, 5) = b(3);
    hess(2, 2) = 2 * b(2);
    hess(3, 3) = 2 * b(3);
    hess(4, 0) = b(2);
    hess(5, 1) = b(3);
  }

  bool HasHessian() const override { return true; }
};

TEST(DynamicsTest, SizeFunctions) {
  TestDynamics<4, 2> model;
  EXPECT_EQ(model.NStates, 4);
  EXPECT_EQ(model.NControls, 2);
  EXPECT_EQ(model.NOutputs, 4);
  EXPECT_EQ(model.StateDimension(), 4);
  EXPECT_EQ(model.ControlDimension(), 2);
  EXPECT_EQ(model.OutputDimension(), 4);
}

TEST(DynamicsTest, Dynamics) {
  TestDynamics<4, 2> model;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);
  VectorXd xdot(4);
  const float t = 0.1;
  model.Evaluate(x, u, t, xdot);
  VectorXd expected(4);
  expected << 5 * t, 6 * t, 5 * 1 + 9, 6 * 2 + 16;
  EXPECT_TRUE(xdot.isApprox(expected, 1e-8));

  model.SetTime(t);
  model.Evaluate(x, u, xdot);
  EXPECT_TRUE(xdot.isApprox(expected, 1e-8));

  FunctionBase* f = &model;
  xdot.setZero();
  f->Evaluate(x, u, xdot);
  EXPECT_TRUE(xdot.isApprox(expected, 1e-6));
}

TEST(DynamicsTest, Jacobian) {
  TestDynamics<4, 2> model;

  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);
  RowMajorXd jac(4, 6);
  const float t = 0.1;
  model.Jacobian(x, u, t, jac);
  RowMajorXd expected(4, 6);
  // clang-format off
  expected << 
      0, 0, 0, 0, t, 0, 
      0, 0, 0, 0, 0, t, 
      u(0), 0, 2 * x(2), 0, x(0), 0, 
      0, u(1), 0, 2 * x(3), 0, x(1);
  // clang-format on
  EXPECT_TRUE(jac.isApprox(expected, 1e-8));

  model.SetTime(t);
  model.Jacobian(x, u, jac);
  EXPECT_TRUE(jac.isApprox(expected, 1e-8));

  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(model.CheckJacobian(1e-4));
  }
}

TEST(DynamicsTest, Hessian) {
  TestDynamics<4, 2> model;

  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);
  VectorXd b = Eigen::Vector4d(-1, -2, -3, -4);
  MatrixXd hess(6, 6);
  hess.setZero();
  const float t = 0.1;
  model.Hessian(x, u, t, b, hess);
  MatrixXd expected(6, 6);
  expected.setZero();
  // clang-format off
  expected << 0, 0, 0, 0, b(2), 0, 0, 0, 0, 0, 0, b(3), 0, 0, 2 * b(2), 0, 0, 0, 0, 0, 0, 2 * b(3),
      0, 0, b(2), 0, 0, 0, 0, 0, 0, b(3), 0, 0, 0, 0;
  // clang-format on
  EXPECT_TRUE(hess.isApprox(expected, 1e-8));

  model.SetTime(t);
  model.Hessian(x, u, b, hess);
  EXPECT_TRUE(hess.isApprox(expected, 1e-8));

  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(model.CheckHessian(1e-4));
  }
}

}  // namespace problem
}  // namespace altro