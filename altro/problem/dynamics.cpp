// Copyright [2021] Optimus Ride Inc.

#include "altro/problem/dynamics.hpp"

namespace altro {
namespace problem {

VectorXd ContinuousDynamics::Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t) {
  VectorXd xdot(x.rows());
  Evaluate(x, u, t, xdot);
  return xdot;
}

VectorXd ContinuousDynamics::operator()(const VectorXdRef& x, const VectorXdRef& u, const float t) {
  return Evaluate(x, u, t);
}

VectorXd DiscreteDynamics::Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                                    const float h) {
  VectorXd xdot(x.rows());
  Evaluate(x, u, t, h, xdot);
  return xdot;
}

VectorXd DiscreteDynamics::operator()(const VectorXdRef& x, const VectorXdRef& u, const float t,
                                      const float h) {
  return Evaluate(x, u, t, h);
}

}  // namespace problem
}  // namespace altro