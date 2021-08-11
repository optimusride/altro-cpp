#include <iostream>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "altro/common/functionbase.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/utils.hpp"

namespace altro {

namespace {

template <class MatA, class MatB>
bool MatrixComparison(const MatA& expected, const MatB& actual, const double eps, const bool verbose) {
  // Compare
  double err = (expected - actual).norm();

  // Print results
  if (verbose) {
    if (err > eps) {
      fmt::print("Calculated:\n{}\n", actual);
      fmt::print("Finite Diff: \n{}\n", expected);
    }
    fmt::print("Error: {}\n", err);
  }
  return err < eps;
}

}  // namespace

bool FunctionBase::CheckJacobian(const double eps, const bool verbose) {
  const int n = StateDimension();
  const int m = ControlDimension();
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  return CheckJacobian(x, u, eps, verbose);
}

bool FunctionBase::CheckJacobian(const VectorXdRef& x, const VectorXdRef& u, 
                    const double eps, const bool verbose) {
  int p = OutputDimension();
  int n = x.size(); 
  int m = u.size(); 

  // NOTE(bjackson): The NOLINT comments here and below are to surpress clang-tidy 
  // warnings about uninitialized values, even though these are clearly initialized.
  MatrixXd fd_jac = MatrixXd::Zero(p, n + m);  // NOLINT
  MatrixXd jac = MatrixXd::Zero(p, n + m);  // NOLINT
  VectorXd z(n + m);
  z << x, u;

  // Calculate Jacobian
  Jacobian(x, u, jac);

  // Calculate using finite differencing
  auto fz = [&](auto z) -> VectorXd { 
    VectorXd out(this->OutputDimension());
    this->Evaluate(z.head(n), z.tail(m), out); 
    return out;
  };
  fd_jac = utils::FiniteDiffJacobian<Eigen::Dynamic, Eigen::Dynamic>(fz, z);

  return MatrixComparison(fd_jac, jac, eps, verbose);
}

bool FunctionBase::CheckHessian(const double eps, const bool verbose) {
  int n = StateDimension();
  int m = ControlDimension();
  int p = OutputDimension();
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  VectorXd b;
  if (p == 1) {
    b.setOnes(p);
  } else {
    b.setRandom(p);
  }
  return CheckHessian(x, u, b, eps, verbose);
}
bool FunctionBase::CheckHessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
                  const double eps, const bool verbose) {
  int n = StateDimension();
  int m = ControlDimension();
  VectorXd z(n + m);
  z << x, u;

  MatrixXd hess = MatrixXd::Zero(n + m, n + m);
  Hessian(x, u, b, hess);

  auto jvp = [&](auto z) -> double {
    VectorXd out(this->OutputDimension());
    this->Evaluate(z.head(n), z.tail(m), out);
    return out.transpose() * b;
  };
  MatrixXd fd_hess = utils::FiniteDiffHessian(jvp, z);

  return MatrixComparison(fd_hess, hess, eps, verbose);
}

bool ScalarFunction::CheckGradient(const double eps, const bool verbose) {
  const int n = this->StateDimension();
  const int m = this->ControlDimension();
  VectorXd x = VectorXd::Random(n);  // NOLINT
  VectorXd u = VectorXd::Random(m);  // NOLINT
  return CheckGradient(x, u, eps, verbose); // NOLINT
}

bool ScalarFunction::CheckGradient(const VectorXdRef& x, const VectorXdRef& u, const double eps, const bool verbose) {
  int n = x.size(); 
  int m = u.size(); 
  VectorXd z(n + m);
  z << x, u;

  VectorXd grad = VectorXd::Zero(n + m);  // NOLINT
  VectorXd fd_grad = VectorXd::Zero(n + m);  // NOLINT
  Gradient(x, u, grad);

  auto fz = [&](auto z) -> double { return this->Evaluate(z.head(n), z.tail(m)); };
  fd_grad = utils::FiniteDiffGradient<-1>(fz, z);

  return MatrixComparison(fd_grad, grad, eps, verbose); // NOLINT
}

}  // namespace altro
