#pragma once

#include "dynamics.hpp"
#include "eigentypes.hpp"

namespace altro {
namespace examples {

/**
 * @brief Triple integrator dynamics model, where the jerk is the control input
 *
 * State vector x = [x1 x2... v1 v2... a1 a2...]
 * Control vector u = [j1 j2...]
 * where xi, vi, ai, and ji are the position, velocity, acceration, and jerk in
 * dimension i.
 *
 */
class TripleIntegrator : public ContinuousDynamics {
 public:
  TripleIntegrator(int dof = 1) : dof_(dof) {
    ALTRO_ASSERT(dof > 0, "The degrees of freedom must be greater than 0.");
  }

  int StateDimension() const override { return 3 * dof_; }
  int ControlDimension() const override { return dof_; }

  VectorXd Evaluate(const VectorXd& x, const VectorXd& u,
                    const float t) const override;
  void Jacobian(const VectorXd& x, const VectorXd& u, const float t,
                MatrixXd& jac) const override;
  void Hessian(const VectorXd& x, const VectorXd& u, const float t,
               const VectorXd& b, MatrixXd& hess) const override;
  bool HasHessian() const override { return true; };

 private:
  int dof_;
};

}  // namespace examples
}  // namespace altro