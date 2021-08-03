#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>

#include "altro/common/functionbase.hpp"
#include "altro/problem/costfunction.hpp"

namespace altro {
namespace problem {

class TestCostFunction : public CostFunction {
 public:
  // Provide access to optional ScalarFunction API
  using ScalarFunction::Gradient;
  using ScalarFunction::Hessian;

  static constexpr int NStates = 4;
  static constexpr int NControls = 2;
  int StateDimension() const override { return NStates; }
  int ControlDimension() const override { return NControls; }
  
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override {
    return x.squaredNorm() + u.squaredNorm();
  }
  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                Eigen::Ref<VectorXd> du) override {
    dx = 2 * x;
    du = 2 * u;
  }
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                       Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    dxdx.setIdentity();
    dudu.setIdentity();
    dxdu.setZero();
  }
};

TEST(FunctionBase, CostFunSizes) {
  TestCostFunction costfun;
  EXPECT_EQ(costfun.NStates, 4);
  EXPECT_EQ(costfun.NControls, 2);
  EXPECT_EQ(costfun.NOutputs, 1);
  EXPECT_EQ(costfun.StateDimension(), 4);
  EXPECT_EQ(costfun.ControlDimension(), 2);
  EXPECT_EQ(costfun.OutputDimension(), 1);
}

TEST(FunctionBase, CostFunEval) {
  TestCostFunction costfun;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);
  double J = costfun.Evaluate(x,u);
  const double J_expected = 1 + 4 + 9 + 16 + 25 + 36;
  EXPECT_DOUBLE_EQ(J, J_expected);
}

TEST(FunctionBase, CostFunGradient) {
  TestCostFunction costfun;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);

  VectorXd dx(costfun.StateDimension());
  VectorXd du(costfun.ControlDimension());
  costfun.Gradient(x, u, dx, du);
  EXPECT_TRUE(dx.isApprox(2 * x));
  EXPECT_TRUE(du.isApprox(2 * u));

  VectorXd grad = VectorXd::Zero(dx.size() + du.size());
  VectorXd grad_expected = grad;
  grad_expected << dx, du;
  costfun.Gradient(x, u, grad);
  EXPECT_TRUE(grad.isApprox(grad_expected));

  for (int i = 0; i < 10; ++i) {
    costfun.CheckGradient();
  }
}

TEST(FunctionBase, CostFunHessian) {
  TestCostFunction costfun;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  VectorXd u = Eigen::Vector2d(5, 6);

  const int n = costfun.StateDimension();
  const int m = costfun.ControlDimension();
  MatrixXd dxdx(n, n);
  MatrixXd dxdu(n, m);
  MatrixXd dudu(m, m);
  costfun.Hessian(x, u, dxdx, dxdu, dudu);
  EXPECT_TRUE(dxdx.isApprox(MatrixXd::Identity(n, n)));
  EXPECT_TRUE(dudu.isApprox(MatrixXd::Identity(m, m)));
  EXPECT_TRUE(dxdu.isApproxToConstant(0));

  MatrixXd hess = MatrixXd::Zero(n + m, n + m);
  MatrixXd hess_expected = hess;
  hess_expected << dxdx, dxdu, dxdu.transpose(), dudu;
  costfun.Hessian(x, u, hess);
  EXPECT_TRUE(hess.isApprox(hess_expected));

  for (int i = 0; i < 10; ++i) {
    costfun.CheckHessian();
  }
}

}  // namespace problem
}  // namespace altro