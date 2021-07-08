#include "examples/unicycle.hpp"

#include <cmath>

#include "altro/utils/utils.hpp"

namespace altro {
namespace examples {

void Unicycle::EvaluateInplace(const VectorXdRef& x, const VectorXdRef& u, const float t,
                               Eigen::Ref<VectorXd> xdot) const {
  ALTRO_UNUSED(t);
  double theta = x(2);  // angle
  double v = u(0);      // linear velocity
  double omega = u(1);  // angular velocity
  xdot(0) = v * cos(theta);
  xdot(1) = v * sin(theta);
  xdot(2) = omega;
}

void Unicycle::Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                        Eigen::Ref<MatrixXd> jac) const {
  ALTRO_UNUSED(t);
  double theta = x(2);  // angle
  double v = u(0);      // linear velocity
  jac(0, 2) = -v * sin(theta);
  jac(0, 3) = cos(theta);
  jac(1, 2) = v * cos(theta);
  jac(1, 3) = sin(theta);
  jac(2, 4) = 1;
}

void Unicycle::Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                       const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) const {
  ALTRO_UNUSED(t);
  double theta = x(2);  // angle
  double v = u(0);      // linear velocity
  hess(2, 2) = -b(0) * v * cos(theta) - b(1) * v * sin(theta);
  hess(2, 3) = -b(0) * sin(theta) + b(1) * cos(theta);
  hess(3, 2) = hess(2, 3);
}

}  // namespace examples
}  // namespace altro