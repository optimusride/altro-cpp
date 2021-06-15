#include "integration.hpp"

namespace altro {

template <class Model, class Integrator = RungeKutta4>
class DiscretizedModel : public DiscreteDynamics 
{
 public:
  DiscretizedModel(const Model& model) : model_(model) {}
  VectorXd Evaluate(const VectorXd& x, const VectorXd& u, 
                    const float t, const float h) const override {
    return integrator_.Integrate(model_, x, u, t, h);
  } 

  void Jacobian(const VectorXd& x, const VectorXd& u, 
                const float t, const float h, MatrixXd& jac) const override {
    integrator_.Jacobian(model_, x, u, t, h, jac);
  } 

  void Hessian(const VectorXd& x, const VectorXd& u, 
               const float t, const float h, 
               const VectorXd& b, MatrixXd& hess) const override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(h);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  bool HasHessian() const override { return model_.HasHessian(); }
  int StateDimension() const override { return model_.StateDimension(); }
  int ControlDimension() const override { return model_.ControlDimension(); }

 private:
  Model model_;
  Integrator integrator_;
};
	
} // namespace altro
