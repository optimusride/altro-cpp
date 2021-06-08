// #include <Eigen/Dense>
#include <Eigen/Dense>

namespace altro {
namespace trajectory {

class ExplicitIntegrator
{
 public:
  using VectorXd = Eigen::VectorXd;
  typedef VectorXd DynamicsFunc(VectorXd, VectorXd, float);
  virtual VectorXd Integrate(DynamicsFunc dynamics, VectorXd x, VectorXd u, float t, float h) = 0;
};

class RungeKutta4 final : public ExplicitIntegrator
{
 public:
  using VectorXd = ExplicitIntegrator::VectorXd;
  using DynamicsFunc = ExplicitIntegrator::DynamicsFunc;
  VectorXd Integrate(DynamicsFunc dynamics, VectorXd x, VectorXd u, float t, float h) override;
};

} // namepsace trajectory
} // namespace altro