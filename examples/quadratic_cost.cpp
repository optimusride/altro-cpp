#include "examples/quadratic_cost.hpp"

namespace altro {
namespace examples {

double QuadraticCost::Evaluate(const VectorXdRef& x,
                               const VectorXdRef& u) const {
  return 0.5 * x.dot(Q_ * x) + x.dot(H_ * u) + 0.5 * u.dot(R_ * u) + q_.dot(x) + r_.dot(u) + c_;
}

void QuadraticCost::Gradient(const VectorXdRef& x,
                             const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                             Eigen::Ref<VectorXd> du) const {
  dx = Q_ * x + q_ + H_ * u;
  du = R_ * u + r_ + H_.transpose() * x;
}

void QuadraticCost::Hessian(const VectorXdRef& x,
                            const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                            Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) const {
  ALTRO_UNUSED(x);
  ALTRO_UNUSED(u);
  dxdx = Q_;
  dudu = R_;
  dxdu = H_;
}

}  // namespace examples
}  // namespace altro