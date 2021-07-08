#pragma once

#include "altro/eigentypes.hpp"

namespace altro {
namespace utils {
/**
 * @brief Calculate the approximate Jacobian of the function @param f using
 * finite-differences
 *
 * @tparam ncols static size of input. Can be -1 for heap-allocated vectors.
 * @tparam Func generic function type (just needs () operator)
 * @tparam nrows static size of the output. Can be -1 (default) for
 * heap-allocated arrays
 * @tparam T floating point type. Should be double-precision for best results.
 * @param f function-like object that implements () operator taking in the
 * vector x and returning a vector y.
 * @param x vector input to the function `func`
 * @param eps perturbation size for finite difference step
 * @param central if `true` will use the more accurate but more computationally
 * expensive central-difference method
 * @return Eigen::Matrix<T, nrows, ncols>
 */
template <class Func>
Eigen::MatrixXd FiniteDiffJacobian(const Func &f,
                                   const VectorXdRef &x,
                                   const double eps = 1e-6,
                                   const bool central = false) {
  return FiniteDiffJacobian<-1, -1, Func>(f, x, eps, central);
}

template <int nrows, int ncols, class Func>
Eigen::Matrix<double, nrows, ncols> FiniteDiffJacobian(
    const Func &f, const Eigen::Ref<const Eigen::Matrix<double, ncols, 1>> &x,
    const double eps = 1e-6, const bool central = false) {
  const int n = x.rows();

  // Evaluate the function and get output size
  Eigen::Matrix<double, nrows, 1> y = f(x);
  const int m = y.rows();
  Eigen::Matrix<double, nrows, ncols> jac =
      Eigen::Matrix<double, nrows, ncols>::Zero(m, n);

  // Create pertubation vector
  Eigen::Matrix<double, ncols, 1> e = Eigen::Matrix<double, ncols, 1>::Zero(n);

  // Loop over columns
  e(0) = eps;
  for (int i = 0; i < n; ++i) {
    double step = eps;
    if (central) {
      y = f(x - e);
      step = 2 * eps;
    }
    jac.col(i) = (f(x + e) - y) / step;
    if (i < n - 1) {
      e(i + 1) = e(i);
      e(i) = 0;
    }
  }
  return jac;
}

/**
 * @brief Converts a function object that returns a scalar to one that returns a
 * 1D vector
 *
 * @tparam Func function-like object
 */
template <class Func>
struct ScalarToVec {
  using Vector1d = Eigen::Matrix<double, 1, 1>;
  Vector1d operator()(const VectorXd &x) const {
    Vector1d y;
    y << f(x);
    return y;
  }
  Func f;
};

/**
 * @brief Calculate the gradient of a scalar-valued function using finite
 * differences
 * Simply converts the function to return a 1D vector and calls
 * `FiniteDiffJacobian`
 *
 * See `FiniteDiffJacobian` for full docstring.
 *
 * @return Eigen::Matrix<T, ncols, 1> a column vector
 */
template <int ncols, class Func>
Eigen::Matrix<double, ncols, 1> FiniteDiffGradient(
    const Func &f, const Eigen::Matrix<double, ncols, 1> &x,
    const double eps = 1e-6, const bool central = false) {
  ScalarToVec<Func> f2 = {f};
  return FiniteDiffJacobian<1, ncols, ScalarToVec<Func>>(f2, x, eps, central)
      .transpose();
}

/**
 * @brief A functor that calculates the gradient of an arbitrary scalar-valued
 * function
 *
 * @tparam nrows static size of input. Can be Eigen::Dynamic.
 * @tparam Func function-like object
 * @tparam T floating-point precision
 */
template <int nrows, class Func, class T>
struct FiniteDiffGradientFunc {
  using GradVec = Eigen::Matrix<T, nrows, 1>;
  GradVec operator()(const GradVec &x) const {
    return FiniteDiffGradient(f, x, eps, central);
  }
  Func f;
  double eps;
  bool central;
};

/**
 * @brief Calculate the hessian of a scalar-valued function f
 * Generates a functor that calculates the gradient using finite-differences and
 * then calls `FiniteDiffJacobian` on that object.
 *
 * See `FiniteDiffJacobian` for full docstring
 *
 * @return Eigen::Matrix<T, ncols, ncols>
 */
template <int ncols, class Func>
Eigen::Matrix<double, ncols, ncols> FiniteDiffHessian(
    const Func &f, const Eigen::Matrix<double, ncols, 1> &x,
    const double eps = 1e-4, const bool central = true) {
  using GradFunc = FiniteDiffGradientFunc<ncols, Func, double>;
  GradFunc gradfun = {f, eps, central};
  return FiniteDiffJacobian<ncols, ncols, GradFunc>(gradfun, x, eps, central);
}

}  // namespace utils
}  // namespace altro
