#include "examples/triple_integrator.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace examples {

void TripleIntegrator::EvaluateInplace(const Eigen::Ref<const VectorXd>& x,
                                       const Eigen::Ref<const VectorXd>& u,
                                       const float t,
                                       Eigen::Ref<VectorXd> xdot) const {
  ALTRO_UNUSED(t);
  for (int i = 0; i < dof_; ++i) {
    xdot[i] = x[i + dof_];
    xdot[i + dof_] = x[i + 2 * dof_];
    xdot[i + 2 * dof_] = u[i];
  }
}

void TripleIntegrator::Jacobian(const Eigen::Ref<const VectorXd>& x,
                                const Eigen::Ref<const VectorXd>& u,
                                const float t, Eigen::Ref<MatrixXd> jac) const {
  ALTRO_UNUSED(x);
  ALTRO_UNUSED(u);
  ALTRO_UNUSED(t);
  jac.setZero();
  for (int i = 0; i < dof_; ++i) {
    jac(i, i + dof_) = 1;
    jac(i + dof_, i + 2 * dof_) = 1;
    jac(i + 2 * dof_, i + 3 * dof_) = 1;
  }
}

void TripleIntegrator::Hessian(const Eigen::Ref<const VectorXd>& x,
                               const Eigen::Ref<const VectorXd>& u,
                               const float t,
                               const Eigen::Ref<const VectorXd>& b,
                               Eigen::Ref<MatrixXd> hess) const {
  ALTRO_UNUSED(x);
  ALTRO_UNUSED(u);
  ALTRO_UNUSED(t);
  ALTRO_UNUSED(b);
  hess.setZero();
}

}  // namespace examples
}  // namespace altro