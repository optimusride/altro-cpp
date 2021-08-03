#pragma once

#include <vector>
#include <array>
#include <memory>

#include "altro/eigentypes.hpp"
#include "altro/problem/dynamics.hpp"
#include "altro/common/state_control_sized.hpp"
namespace altro {
namespace problem {
/**
 * @brief Interface class for explicit integration methods for dynamical
 * systems.
 *
 * All sub-classes must implement the `Integrate` method that integrates an
 * arbitrary functor over some time step, as well as it's first derivative
 * via the `Jacobian` method. 
 * 
 * Sub-classes should have a constructor that takes the state and control 
 * dimension, e.g.:
 * 
 * `MyIntegrator(int n, int m);`
 *
 * @tparam DynamicsFunc the type of the function-like object that evaluates a
 * first-order ordinary differential equation with the following signature:
 * dynamics(const VectorXd& x, const VectorXd& u, float t) const
 *
 * See `ContinuousDynamics` class for the expected interface.
 */
template <int NStates, int NControls>
class ExplicitIntegrator : public StateControlSized<NStates, NControls> {
 protected:
  using DynamicsPtr = std::shared_ptr<ContinuousDynamics>;

 public:
  ExplicitIntegrator(int n, int m) : StateControlSized<NStates, NControls>(n, m) {}
  ExplicitIntegrator() : StateControlSized<NStates, NControls>() {

  }
  virtual ~ExplicitIntegrator() = default;

  /**
   * @brief Integrate the dynamics over a given time step
   *
   * @param[in] dynamics ContinuousDynamics object to evaluate the continuous
   * dynamics
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h discretization step length (e.g. time step)
   * @return VectorXd state vector at the end of the time step
   */
  virtual void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x,
                         const VectorXdRef& u, float t, float h,
                         Eigen::Ref<VectorXd> xnext) = 0;

  /**
   * @brief Evaluate the Jacobian of the discrete dynamics
   *
   * Will typically call the continuous dynamics Jacobian.
   *
   * @pre Jacobian must be initialized
   *
   * @param[in] dynamics ContinuousDynamics object to evaluate the continuous
   * dynamics
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h discretization step length (e.g. time step)
   * @param[out] jac discrete dynamics Jacobian evaluated at x, u, t.
   */
  virtual void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x,
                        const VectorXdRef& u, float t, float h, JacobianRef jac) = 0;
};

/**
 * @brief Basic explicit Euler integration
 *
 * Simplest integrator that requires only a single evaluationg of the continuous
 * dynamics but suffers from significant integration errors.
 *
 * @tparam DynamicsFunc
 */
class ExplicitEuler final : public ExplicitIntegrator<Eigen::Dynamic, Eigen::Dynamic> {
 public:
  ExplicitEuler(int n, int m) : ExplicitIntegrator<Eigen::Dynamic, Eigen::Dynamic>(n, m) {}
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                 float t, float h, Eigen::Ref<VectorXd> xnext) override {
    dynamics->Evaluate(x, u, t,  xnext);
    xnext = x + xnext * h;
  }
  void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                float t, float h, JacobianRef jac) override {
    int n = x.size();
    int m = u.size();
    dynamics->Jacobian(x, u, t, jac);
    jac = MatrixXd::Identity(n, n + m) + jac * h;
  }
};

/**
 * @brief Fourth-order explicit Runge Kutta integrator.
 *
 * De-facto explicit integrator for many robotics applications.
 * Good balance between accuracy and computational effort.
 *
 * @tparam DynamicsFunc
 */
template <int NStates, int NControls>
class RungeKutta4 final : public ExplicitIntegrator<NStates, NControls> {
  using typename ExplicitIntegrator<NStates, NControls>::DynamicsPtr;
 public:

  RungeKutta4(int n, int m) : ExplicitIntegrator<NStates, NControls>(n, m) {
    Init();
  }
  RungeKutta4() : ExplicitIntegrator<NStates, NControls>() {
    Init();
  }
  void Integrate(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                 float t, float h, Eigen::Ref<VectorXd> xnext) override {

    dynamics->Evaluate(x, u, t, k1_);
    dynamics->Evaluate(x + k1_ * 0.5 * h, u, t + 0.5 * h, k2_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k2_ * 0.5 * h, u, t + 0.5 * h, k3_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k3_ * h, u, t + h, k4_);
    xnext = x + h * (k1_ + 2 * k2_ + 2 * k3_ + k4_) / 6;  // NOLINT(readability-magic-numbers)
  }
  void Jacobian(const DynamicsPtr& dynamics, const VectorXdRef& x, const VectorXdRef& u,
                float t, float h, JacobianRef jac) override {
    int n = dynamics->StateDimension();
    int m = dynamics->ControlDimension();

    dynamics->Evaluate(x, u, t, k1_);
    dynamics->Evaluate(x + k1_ * 0.5 * h, u, t + 0.5 * h, k2_);  // NOLINT(readability-magic-numbers)
    dynamics->Evaluate(x + k2_ * 0.5 * h, u, t + 0.5 * h, k3_);  // NOLINT(readability-magic-numbers)

    dynamics->Jacobian(x, u, t, jac);
    A_[0] = jac.topLeftCorner(n, n);
    B_[0] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + 0.5 * k1_ * h, u, 0.5 * t, jac);  // NOLINT(readability-magic-numbers)
    A_[1] = jac.topLeftCorner(n, n);
    B_[1] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + 0.5 * k2_ * h, u, 0.5 * t, jac);  // NOLINT(readability-magic-numbers)
    A_[2] = jac.topLeftCorner(n, n);
    B_[2] = jac.topRightCorner(n, m);
    dynamics->Jacobian(x + k3_ * h, u, t, jac);
    A_[3] = jac.topLeftCorner(n, n);
    B_[3] = jac.topRightCorner(n, m);

    dA_[0] = A_[0] * h;
    dA_[1] = A_[1] * (MatrixXd::Identity(n, n) + 0.5 * dA_[0]) * h;  // NOLINT(readability-magic-numbers)
    dA_[2] = A_[2] * (MatrixXd::Identity(n, n) + 0.5 * dA_[1]) * h;  // NOLINT(readability-magic-numbers)
    dA_[3] = A_[3] * (MatrixXd::Identity(n, n) + dA_[2]) * h;

    dB_[0] = B_[0] * h;
    dB_[1] = B_[1] * h + 0.5 * A_[1] * dB_[0] * h;  // NOLINT(readability-magic-numbers)
    dB_[2] = B_[2] * h + 0.5 * A_[2] * dB_[1] * h;  // NOLINT(readability-magic-numbers)
    dB_[3] = B_[3] * h + A_[3] * dB_[2] * h;

    jac.topLeftCorner(n, n) =
        MatrixXd::Identity(n, n)
        + (dA_[0] + 2 * dA_[1] + 2 * dA_[2] + dA_[3]) / 6;  // NOLINT(readability-magic-numbers)
    jac.topRightCorner(n, m) =
        (dB_[0] + 2 * dB_[1] + 2 * dB_[2] + dB_[3]) / 6;  // NOLINT(readability-magic-numbers)
  }

 private:
  void Init() {
    int n = this->StateDimension();
    int m = this->ControlDimension();
    k1_.setZero(n);
    k2_.setZero(n);
    k3_.setZero(n);
    k4_.setZero(n);
    for (int i = 0; i < 4; ++i) {
      A_[i].setZero(n, n); 
      B_[i].setZero(n, m);
      dA_[i].setZero(n, n); 
      dB_[i].setZero(n, m);
    }
  }

  // These need to be mutable to keep the integration methods as const methods
  // Since they replace arrays that would otherwise be created temporarily and 
  // provide no public access, it should be fine.
  VectorNd<NStates> k1_;
  VectorNd<NStates> k2_;
  VectorNd<NStates> k3_;
  VectorNd<NStates> k4_;
  std::array<MatrixNxMd<NStates, NStates>, 4> A_;
  std::array<MatrixNxMd<NStates, NControls>, 4> B_;
  std::array<MatrixNxMd<NStates, NStates>, 4> dA_;
  std::array<MatrixNxMd<NStates, NControls>, 4> dB_;
};

}  // namespace problem
}  // namespace altro
