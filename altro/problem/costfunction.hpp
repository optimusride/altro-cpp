// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>

#include "altro/eigentypes.hpp"
#include "altro/common/functionbase.hpp"
#include "altro/utils/derivative_checker.hpp"

namespace altro {
namespace problem {

/**
 * @brief Represents a scalar-valued cost function.
 *
 * As a specialization of the ScalarFunction interface, users are expected to
 * implement the interface described below. The key difference from the
 * `ScalarFunction` interface is that the partial derivatives with repsect to the
 * states and controls are passed in separately instead of as a single argument.
 * The original API simply passes in the appropriate portions of the joint
 * derivative.
 *
 * # Interface
 * The user must define the following functions:
 * - `int StateDimension() const` - number of states (length of x)
 * - `int ControlDimension() const` - number of controls (length of u)
 * - `double Evaluate(const VectorXdRef& x, const VectorXdRef& u)`
 * - `void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> dx,
 * Eigen::Ref<Eigen::VectorXd> du)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::MatrixXd> dxdx,
 * Eigen::Ref<Eigen::MatrixXd> dxdu, Eigen::Ref<Eigen::MatrixXd> dudu)`
 * - `bool HasHessian() const` - Specify if the Hessian is implemented - optional (assumed to be
 * true)
 * 
 * Where we use the following Eigen type alias:
 *    using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * The user also has the option of defining the static constants:
 *    static constexpr int NStates
 *    static constexpr int NControls
 *
 * which can be used to provide compile-time size information. For best performance, 
 * it is highly recommended that the user specifies these constants for their implementation.
 * 
 * # ScalarFunction API
 * To use the ScalarFunction API, insert the following lines into the public 
 * interface of the derived class:
 *    using ScalarFunction::Gradient;
 *    using ScalarFunction::Hessian;
 */
class CostFunction : public altro::ScalarFunction {
 public:
  using altro::ScalarFunction::Hessian;

  // New Interface
  virtual void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                        Eigen::Ref<VectorXd> du) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                       Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) = 0;

  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> grad) override {
    Gradient(x, u, grad.head(StateDimension()), grad.tail(ControlDimension()));
  }
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> hess) override {
    const int n = StateDimension();
    const int m = ControlDimension();
    constexpr int Nx = NStates;
    constexpr int Nu = NControls;
    Hessian(x, u, hess.topLeftCorner<Nx, Nx>(n, n), hess.topRightCorner<Nx, Nu>(n, m),
            hess.bottomRightCorner<Nu, Nu>(m, m));
  }
};
}  // namespace problem
}  // namespace altro