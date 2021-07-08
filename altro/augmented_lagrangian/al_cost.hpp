#pragma once

#include <memory>

#include "altro/problem/costfunction.hpp"

namespace altro {
namespace augmented_lagrangian {

/**
 * @brief An augmented Lagrangian cost function, which adds the linear and
 * quadratic penalty costs to an existing cost function.
 * 
 */
class ALCost : public problem::CostFunction {
 public:
  double Evaluate(const VectorXdRef& x,
                  const VectorXdRef& u) const override {
    double J = costfun_->Evaluate(x, u);
    // J += lambda'c + 0.5 * c'c
    return J;
  }

  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) const override {
    costfun_->Gradient(x, u, dx, du);
		// conset_->AugLagGradient(x, u, dx, du);
    // dx += Cx'(lambda + mu * c)
    // du += Cu'(lambda + mu * c)
  }

  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u,
                       Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
                       Eigen::Ref<MatrixXd> dudu) const override {
    costfun_->Hessian(x, u, dxdx, dxdu, dudu);
		// conset_->ALHessian(x, u, dxdx, dxdu, dudu, full_newton_);
		// dxdx += mu * Cx'Cx 
		// dxdu += mu * Cx'Cu 
		// dudu += mu * Cu'Cu 
		if (full_newton_) {
			// dxdx += Cxx'(lambda + mu * c)
			// dxdu += Cxu'(lambda + mu * c)
			// dudu += Cuu'(lambda + mu * c)
		}
  }

 private:
  std::shared_ptr<problem::CostFunction> costfun_;
	bool full_newton_ = false;
};

}  // namespace augmented_lagrangian
}  // namespace