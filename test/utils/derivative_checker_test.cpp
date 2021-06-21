#include <gtest/gtest.h>
#include <iostream>

#include "altro/utils/derivative_checker.hpp"

namespace altro {
namespace utils {

struct TestFunc {
  VectorXd operator()(const VectorXd& x) const {
    VectorXd out = VectorXd(5);
    out << sin(x(1)), cos(x(2)), x(3) * exp(x(0)), pow(x(0), 2) * 10,
        x(3) * x(2) + sin(pow(x(1), 2));
    return out;
  }
};

TEST(DerivativeCheckerTest, Jacobian) {
  TestFunc f;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  MatrixXd jac = FiniteDiffJacobian<5, 4>(f, x);

  MatrixXd ans = MatrixXd(5, 4);
  ans << 0, cos(x(1)), 0, 0, 0, 0, -sin(x(2)), 0, x(3) * exp(x(0)), 0, 0,
      exp(x(0)), x(0) * 20, 0, 0, 0, 0, cos(pow(x(1), 2)) * 2 * x(1), x(3),
      x(2);

  EXPECT_TRUE(jac.isApprox(ans, 1e-6));

  Eigen::Vector4d x2(x);
  auto jac2 = FiniteDiffJacobian<5, 4, TestFunc>(f, x2);
  EXPECT_TRUE(jac2.isApprox(ans, 1e-6));

  auto jac3 = FiniteDiffJacobian<5, 4, TestFunc>(f, x2);
  EXPECT_TRUE(jac3.isApprox(ans, 1e-6));
}

TEST(DerivativeCheckerTest, JacobianCentralDiff) {
  TestFunc f;
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  double eps = 1e-6;
  MatrixXd jac = FiniteDiffJacobian<5,4>(f, x);

  MatrixXd ans = MatrixXd(5, 4);
  ans << 0, cos(x(1)), 0, 0, 0, 0, -sin(x(2)), 0, x(3) * exp(x(0)), 0, 0,
      exp(x(0)), x(0) * 20, 0, 0, 0, 0, cos(pow(x(1), 2)) * 2 * x(1), x(3),
      x(2);

  double err_forward = (jac - ans).norm();

  bool central = true;
  MatrixXd jac2 = FiniteDiffJacobian<5,4>(f, x, eps, central);
  double err_central = (jac2 - ans).norm();
  // std::cout << "forward: " << err_forward << std::endl
  //           << "central: " << err_central << std::endl;
  EXPECT_GT(log10(err_forward / err_central), 2);
}

TEST(DerivativeCheckerTest, JacobianLambda) {
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  auto f2 = [](auto x) -> Eigen::Vector3d {
    return Eigen::Vector3d(sin(x(0)), pow(sin(x(1)), 2) * x(3), x(3) * x(2));
  };

  auto jac = FiniteDiffJacobian(f2, x);
  MatrixXd ans(3, 4);
  ans << cos(x(0)), 0, 0, 0, 0, sin(x(1)) * 2 * cos(x(1)) * x(3), 0,
      pow(sin(x(1)), 2), 0, 0, x(3), x(2);
  EXPECT_TRUE(jac.isApprox(ans, 1e-6));

  Eigen::Vector4d x2(x);
  auto jac2 = FiniteDiffJacobian(f2, x);
  EXPECT_TRUE(jac2.isApprox(ans, 1e-6));
}

struct TestFuncScalar {
  double operator()(const VectorXd& x) const {
    return x(0) * x(1) + sin(x(2) * x(3)) * x(3);
  }
};

TEST(DerivativeCheckerTest, Gradient) {
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  TestFuncScalar f;
  auto grad = FiniteDiffGradient(f, x);
  VectorXd ans(4);
  ans << x(1), x(0), cos(x(2) * x(3)) * x(3) * x(3),
      cos(x(2) * x(3)) * x(3) * x(2) + sin(x(2) * x(3));
  EXPECT_TRUE(ans.isApprox(grad, 1e-5));
  double err = (ans - grad).norm();

  auto grad2 = FiniteDiffGradient(f, x, 1e-10);
  double err2 = (ans - grad2).norm();
  EXPECT_TRUE(ans.isApprox(grad2, 1e-6));
  EXPECT_LT(err2, err);
}

TEST(DerivativeCheckerTest, GradientLambda) {
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  auto f = [](auto x) -> double { return x(0) * x(1) + cos(x(2) * exp(x(3))); };
  Eigen::Vector4d ans(x(1), x(0), -sin(x(2) * exp(x(3))) * exp(x(3)),
                      -sin(x(2) * exp(x(3))) * x(2) * exp(x(3)));
  auto grad = FiniteDiffGradient(f, x, 1e-10);
  EXPECT_TRUE(grad.isApprox(ans, 1e-6));
}

TEST(DerivativeCheckerTest, GradientFunc) {
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  double eps = 1e-6;

  TestFuncScalar f;
  Eigen::Matrix<double, 4, 1> grad = FiniteDiffGradient(f, x, eps);
  bool central = false;
  FiniteDiffGradientFunc<4, TestFuncScalar, double> gradfun = {f, eps, central};
  EXPECT_TRUE(grad.isApprox(gradfun(x)));

  central = true;
  Eigen::Matrix<double, 4, 1> grad2 = FiniteDiffGradient(f, x, eps, central);
  FiniteDiffGradientFunc<4, TestFuncScalar, double> gradfun2 = {f, eps,
                                                                central};
  EXPECT_TRUE(grad2.isApprox(gradfun2(x)));

  // Eigen::Vector4d x2(x);
  // std::cout << gradfun(x2) << std::endl;
}

TEST(DerivativeCheckerTest, Hessian) {
  VectorXd x = Eigen::Vector4d(1, 2, 3, 4);
  TestFuncScalar f;
  double eps = 1e-4;
  double central = false;
  auto hess = FiniteDiffHessian(f, x, eps, central);
  central = true;
  auto hess2 = FiniteDiffHessian(f, x, eps, central);

  MatrixXd ans(4, 4);
  double dx3dx4 =
      -sin(x(2) * x(3)) * x(3) * x(3) * x(2) + cos(x(2) * x(3)) * 2 * x(3);
  ans << 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -sin(x(2) * x(3)) * pow(x(3), 3), dx3dx4,
      0, 0, dx3dx4, -sin(x(2) * x(3)) * pow(x(2), 2) * x(3) +
                        cos(x(2) * x(3)) * x(2) + cos(x(2) * x(3)) * x(2);

  // Check with static size
  Eigen::Vector4d x2(x);
  auto hess3 = FiniteDiffHessian(f, x2, eps, central);
  EXPECT_TRUE(hess3.isApprox(hess2));

  // Check errors w/ central diffing
  double err_forward = (hess - ans).norm();
  double err_central = (hess2 - ans).norm();
  // std::cout << "forward: " << err_forward << std::endl
  //           << "central: " << err_central << std::endl;
  EXPECT_GT(log10(err_forward / err_central), 2);
}

TEST(DerivativeCheckerTest, NewJacobian) {
  TestFunc f;
  VectorXd x2 = VectorXd::LinSpaced(6,1,6);
  VectorXd x = VectorXd::LinSpaced(4,1,4);
  Eigen::Vector4d x3 = VectorXd::LinSpaced(4,1,4);
  MatrixXd jac1 = FiniteDiffJacobian<5, 4, TestFunc>(f, x);
  MatrixXd jac2 = FiniteDiffJacobian<5, 4, TestFunc>(f, x2.head(4));
  MatrixXd jac3 = FiniteDiffJacobian<5, 4, TestFunc>(f, x3);

  MatrixXd ans = MatrixXd(5, 4);
  ans << 0, cos(x(1)), 0, 0, 0, 0, -sin(x(2)), 0, x(3) * exp(x(0)), 0, 0,
      exp(x(0)), x(0) * 20, 0, 0, 0, 0, cos(pow(x(1), 2)) * 2 * x(1), x(3),
      x(2);

  EXPECT_TRUE(jac1.isApprox(ans, 1e-6));
  EXPECT_TRUE(jac2.isApprox(ans, 1e-6));
  EXPECT_TRUE(jac3.isApprox(ans, 1e-6));
}

}  // namespace utils 
}  // namespace altro
