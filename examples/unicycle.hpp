#pragma once

#include "altro/eigentypes.hpp"
#include "altro/problem/dynamics.hpp"

namespace altro {
namespace examples {

/**
 * @brief Simple kinematic model of a unicycle / differential drive robot
 *
 * Has 3 states and 2 controls.
 *
 * # Mathematical Formulation
 * \f[
 * \begin{bmatrix} \dot{x} \\ \dot{y} \dot{\theta}  \end{bmatrix} =
 * \begin{bmatrix} v \cos(\theta) \\ v \sin(\theta) \\ \omega \end{bmatrix}
 * \f]
 *
 * where
 * \f[
 * u = \begin{bmatrix} v \\ \omega \end{bmatrix}
 * \f]
 */
class Unicycle : public problem::ContinuousDynamics {
 public:
  Unicycle() = default;
  static constexpr int NStates = 3;
  static constexpr int NControls = 2;
  int StateDimension() const override { return NStates; }
  int ControlDimension() const override { return NControls; }

  void EvaluateInplace(const VectorXdRef& x, const VectorXdRef& u, const float t,
                       Eigen::Ref<VectorXd> xdot) const override;
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                Eigen::Ref<MatrixXd> jac) const override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) const override;
  bool HasHessian() const override { return true; };
};

}  // namespace examples
}  // namespace altro