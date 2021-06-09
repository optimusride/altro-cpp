#include "quadraticcost.hpp"

namespace altro {
namespace examples {

double QuadraticCost::Evaluate(const VectorXd& x, const VectorXd& u) const
{
  return x.dot(Q_ * x) + x.dot(H_ * u) + u.dot(R_ * u) + q_.dot(x) + r_.dot(u) + c_;
}


void QuadraticCost::Gradient(const VectorXd& x, const VectorXd& u, VectorXd& dx, VectorXd& du) const
{
  dx = Q_ * x + q_;
  du = R_ * u + r_;
}

void QuadraticCost::Hessian(const VectorXd& x, const VectorXd& u, 
               MatrixXd& dxdx, MatrixXd& dxdu, MatrixXd& dudu) const
{
  (void) x;
  (void) u;
  dxdx = Q_;
  dudu = R_;
  dxdu = H_;
}

} // namespace examples
} // namespace altro