#include "altro/problem/dynamics.hpp"

namespace altro {
namespace problem {

template <int Nx, int Nu>
class TestDynamics : public ContinuousDynamics {
 public:
  using FunctionBase::Evaluate;
  using FunctionBase::Jacobian;
  using FunctionBase::Hessian;

  static constexpr int NStates = Nx;
  static constexpr int NControls = Nu;
  static constexpr int NOutputs = Nx;

  int StateDimension() const override { return 4; }
  int ControlDimension() const override { return 2; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, const float t,
                Eigen::Ref<VectorXd> out) override {
    out(0) = u(0) * t;
    out(1) = u(1) * t;
    out(2) = u(0) * x(0) + std::pow(x(2), 2);
    out(3) = u(1) * x(1) + std::pow(x(3), 2);
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, const float t,
                JacobianRef jac) override {
    jac.setZero();
    jac(0, 4) = t;
    jac(1, 5) = t;
    jac(2, 0) = u(0);
    jac(2, 2) = 2 * x(2);
    jac(2, 4) = x(0);
    jac(3, 1) = u(1);
    jac(3, 3) = 2 * x(3);
    jac(3, 5) = x(1);
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const float t, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    hess(0, 4) = b(2);
    hess(1, 5) = b(3);
    hess(2, 2) = 2 * b(2);
    hess(3, 3) = 2 * b(3);
    hess(4, 0) = b(2);
    hess(5, 1) = b(3);
  }

  bool HasHessian() const override { return true; }
};

// Needed to prevent linker errors
template <int Nx, int Nu>
constexpr int TestDynamics<Nx, Nu>::NStates;

template <int Nx, int Nu>
constexpr int TestDynamics<Nx, Nu>::NControls;

template <int Nx, int Nu>
constexpr int TestDynamics<Nx, Nu>::NOutputs;


}  // namespace problem
}  // namespace altro