#pragma once

#include "altro/problem/dynamics.hpp"
#include "altro/eigentypes.hpp"

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
  int StateDimension() const override { return 3; }
  int ControlDimension() const override { return 2; }

  void EvaluateInplace(const Eigen::Ref<const VectorXd>& x,
                       const Eigen::Ref<const VectorXd>& u, const float t,
                       Eigen::Ref<VectorXd> xdot) const override;
  void Jacobian(const Eigen::Ref<const VectorXd>& x,
                const Eigen::Ref<const VectorXd>& u, const float t,
                Eigen::Ref<MatrixXd> jac) const override;
  void Hessian(const Eigen::Ref<const VectorXd>& x,
               const Eigen::Ref<const VectorXd>& u, const float t,
               const Eigen::Ref<const VectorXd>& b,
               Eigen::Ref<MatrixXd> hess) const override;
  bool HasHessian() const override { return true; };
};

}  // namespace examples
}  // namespace altro