#include "integration.hpp"

namespace altro {
namespace trajectory {

using DynamicsFunc = ExplicitIntegrator::DynamicsFunc;
VectorXd RungeKutta4::Integrate(DynamicsFunc dynamics, VectorXd x, VectorXd u, float t, float h) 
{
  VectorXd k1 = dynamics(x, u, t) * h;
  VectorXd k2 = dynamics(x + k1 * 0.5, u, t) * h;
  VectorXd k3 = dynamics(x + k2 * 0.5, u, t) * h;
  VectorXd k4 = dynamics(x + k3, u, t) * h;
  VectorXd xnext = x + (k1 + 2*k2 + 2*k3 + k4) / 6;
  return xnext;
}

} // namepsace trajectory
} // namespace altro