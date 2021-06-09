#pragma once

#include "eigentypes.hpp"
namespace altro {
namespace trajectory {

class ExplicitIntegrator
{
 public:
  typedef VectorXd DynamicsFunc(VectorXd, VectorXd, float);

  virtual ~ExplicitIntegrator() {};
  virtual VectorXd Integrate(DynamicsFunc dynamics, VectorXd x, VectorXd u, float t, float h) = 0;
};

class RungeKutta4 final : public ExplicitIntegrator
{
 public:
  using DynamicsFunc = ExplicitIntegrator::DynamicsFunc;
  VectorXd Integrate(DynamicsFunc dynamics, VectorXd x, VectorXd u, float t, float h) override;
};

} // namepsace trajectory
} // namespace altro