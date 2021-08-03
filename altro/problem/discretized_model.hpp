#pragma once

#include "altro/problem/integration.hpp"

namespace altro {
namespace problem {

/**
 * @brief Discretizes a continuous dynamics model using an explicit integrator.
 * 
 * Uses a specified integrator to integrate a continuous time dynamics model over a 
 * discrete time step.
 * 
 * @tparam Model Model to be discretized. Must inherit from FunctionBase.
 * @tparam Integrator An explicit integrator. Should inherit from ExplicitIntegrator.
 * 
 * For best performance, `Model::NStates` and `Model::NControls` should provide
 * compile-time information about the number of states and controls. This will 
 * allow the integrator to allocate memory on the stack for any temporary arrays
 * needed during the integration procedure.
 */
template <class Model, class Integrator = RungeKutta4<Model::NStates, Model::NControls>>
class DiscretizedModel : public DiscreteDynamics {
 public:
  static_assert(std::is_base_of<FunctionBase, Model>::value, "Model must inherit from FunctionBase.");
  using DiscreteDynamics::Evaluate;

  static constexpr int NStates = Model::NStates;
  static constexpr int NControls = Model::NControls;

  explicit DiscretizedModel(const Model& model)
      : model_(std::make_shared<Model>(model)), integrator_(model.StateDimension(), model.ControlDimension()) {}

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t, const float h,
                       Eigen::Ref<VectorXd> xnext) override {
    integrator_.Integrate(model_, x, u, t, h, xnext);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t, const float h,
                JacobianRef jac) override {
    integrator_.Jacobian(model_, x, u, t, h, jac);
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t, const float h,
               const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(h);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  bool HasHessian() const override { return model_->HasHessian(); }
  int StateDimension() const override { return model_->StateDimension(); }
  int ControlDimension() const override { return model_->ControlDimension(); }

  Integrator& GetIntegrator() { return integrator_; }

 private:
  std::shared_ptr<Model> model_;
  Integrator integrator_;
};

}  // namespace problem
}  // namespace altro
