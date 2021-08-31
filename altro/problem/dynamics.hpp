// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <iostream>

#include "altro/common/functionbase.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace problem {

// clang-format off
/**
 * @brief Represents a continuous dynamics function of the form:
 * \f[ \dot{x} = f(x, u) \f]
 *
 * As a specialization of the `FunctionsBase` interface, the user is
 * expected is expected implement the following interface:
 *
 * # Interface
 * - `int StateDimension() const` - number of states (length of x)
 * - `int ControlDimension() const` - number of controls (length of u)
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, Eigen::Ref<Eigen::VectorXd>
 * out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, Eigen::Ref<MatrixXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, const VectorXdRef& b,
 * Eigen::Ref<Eigen::MatrixXd> hess)` - optional
 * - `bool HasHessian() const` - Specify if the Hessian is implemented
 *
 * Where we use the following Eigen type alias:
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * The user also has the option of defining the static constants:
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *      static constexpr int NOutputs
 *
 * which can be used to provide compile-time size information. For best performance, 
 * it is highly recommended that the user specify these constants, which default to 
 * `Eigen::Dynamic` if not specified.
 *
 * ## FunctionBase API
 * If the original FunctionBase API is needed, the following lines need to be
 * added to the public interface of the derived class:
 *    using FunctionBase::Evaluate;
 *    using FunctionBase::Jacobian;
 *    using FunctionBase::Hessian;
 *
 * NOTE: If using the FunctionBase API with time-varying dynamics, remember
 * that the time must be updated using `ContinuousDynamics::SetTime` before calling
 * `FunctionBase::Evaluate`.
 */
// clang-format on
class ContinuousDynamics : public FunctionBase {
 public:
  using FunctionBase::Evaluate;
  using FunctionBase::Jacobian;
  using FunctionBase::Hessian;

  int OutputDimension() const override { return StateDimension(); }

  // New Interface
  virtual void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t,
                        Eigen::Ref<VectorXd> xdot) = 0;
  virtual void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, Eigen::Ref<MatrixXd> jac) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, const VectorXdRef& b,
                       Eigen::Ref<MatrixXd> hess) = 0;

  // Convenience methods
  VectorXd Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t);
  VectorXd operator()(const VectorXdRef& x, const VectorXdRef& u, float t);

  // FunctionBase API
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) override {
    Evaluate(x, u, GetTime(), out);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    Jacobian(x, u, GetTime(), jac);
  }
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {  // NOLINT(performance-unnecessary-value-param)
    Hessian(x, u, GetTime(), b, hess);
  }

  float GetTime() const { return t_; }
  void SetTime(float t) { t_ = t; }

 protected:
  float t_ = 0.0F;
};

// clang-format off
/**
 * @brief Represents a discrete dynamics function of the form:
 * \f$ x_{k+1} = f(x_k, u_k) \f$
 * 
 * This is the form of the dynamics expected by the altro library. A continuous 
 * time dynamics model can be converted to a discrete model using e.g. `DiscretizedDynamics`.
 *
 * As a specialization of the `FunctionsBase` interface, the user is
 * expected is expected implement the following interface:
 *
 * # Interface
 * - `int StateDimension() const` - number of states (length of x)
 * - `int ControlDimension() const` - number of controls (length of u)
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h, Eigen::Ref<Eigen::VectorXd>
 * out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, Eigen::Ref<MatrixXd> out)`
 * - `void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, const VectorXdRef& b,
 * Eigen::Ref<Eigen::MatrixXd> hess)` - optional
 * - `bool HasHessian() const` - Specify if the Hessian is implemented
 *
 * Where `t` is the time (for time-dependent dynamics) and `h` is the time step. 
 * These can be set and retrieved using `SetTime`, `SetStep`, `GetTime`, and `GetStep`.
 * 
 * Where we use the following Eigen type alias:
 * 
 *      using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *
 * The user also has the option of defining the static constants:
 * 
 *      static constexpr int NStates
 *      static constexpr int NControls
 *      static constexpr int NOutputs
 *
 * which can be used to provide compile-time size information. For best performance, 
 * it is highly recommended that the user specify these constants, which default to 
 * `Eigen::Dynamic` if not specified.
 *
 * ## FunctionBase API
 * If the original FunctionBase API is needed, the following lines need to be
 * added to the public interface of the derived class:
 * 
 *    using FunctionBase::Evaluate;
 *    using FunctionBase::Jacobian;
 *    using FunctionBase::Hessian;
 *
 * NOTE: If using the FunctionBase API with time-varying dynamics, remember
 * that the time must be updated using `DiscreteDynamics::SetTime`  and 
 * `DiscreteDynamics.SetStep` before calling `FunctionBase::Evaluate`.
 */
// clang-format on
class DiscreteDynamics : public FunctionBase {
 public:
  using FunctionBase::Evaluate;
  using FunctionBase::Jacobian;
  using FunctionBase::Hessian;

  int OutputDimension() const override { return StateDimension(); }

  // New Interface
  virtual void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                        Eigen::Ref<VectorXd> xdot) = 0;
  virtual void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, Eigen::Ref<MatrixXd> jac) = 0;
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, const VectorXdRef& b,
                       Eigen::Ref<MatrixXd> hess) = 0;

  // Convenience methods
  VectorXd Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h);
  VectorXd operator()(const VectorXdRef& x, const VectorXdRef& u, float t, float h);

  // FunctionBase API
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> out) override {
    Evaluate(x, u, GetTime(), GetStep(), out);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    Jacobian(x, u, GetTime(), GetStep(), jac);
  }
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override {  // NOLINT(performance-unnecessary-value-param)
    Hessian(x, u, GetTime(), GetStep(), b, hess);
  }

  float GetTime() const { return t_; }
  void SetTime(float t) { t_ = t; }
  float GetStep() const { return h_; }
  void SetStep(float h) { h_ = h; }

 protected:
  float t_ = 0.0F;
  float h_ = 0.0F;
};

}  // namespace problem
}  // namespace altro