#pragma once

#include <memory>

#include "altro/common/functionbase.hpp"
#include "altro/eigentypes.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace constraints {

// Forward-declare for use in ZeroCone
class IdentityCone;

/**
 * @brief An Equality constraint (alias for ZeroCone)
 *
 * Generic equality constraint of the form
 * \f[ g(x,u) = 0 \f]
 *
 * The projection operation for equality constraints of this form projects the
 * value(s) to zero. The dual cone is the identity map.
 */
class ZeroCone {
 public:
  ZeroCone() = delete;
  using DualCone = IdentityCone;

  static void Projection(const VectorXdRef& x, Eigen::Ref<VectorXd> x_proj) {
    ALTRO_ASSERT(x.size() == x_proj.size(), "x and x_proj must be the same size");
    ALTRO_UNUSED(x);
    x_proj.setZero();
  }
  static void Jacobian(const VectorXdRef& x, Eigen::Ref<MatrixXd> jac) {
    ALTRO_UNUSED(x);
    jac.setZero();
  }
  static void Hessian(const VectorXdRef& x, const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
    ALTRO_ASSERT(hess.rows() == hess.cols(), "Hessian must be square.");
    ALTRO_ASSERT(x.size() == b.size(), "x and b must be the same size.");
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(b);
    hess.setZero();
  }
};
using Equality = ZeroCone;

/**
 * @brief The Identity projection
 *
 * The identity projection projects a point onto itself. It is the dual cone
 * for equality constraints, and is used in conic augmented Lagrangian to
 * handle the equality constraints.
 *
 */
class IdentityCone {
 public:
  IdentityCone() = delete;
  using DualCone = ZeroCone;

  static void Projection(const VectorXdRef& x, Eigen::Ref<VectorXd> x_proj) {
    ALTRO_ASSERT(x.size() == x_proj.size(), "x and x_proj must be the same size");
    x_proj = x;
  }
  static void Jacobian(const VectorXdRef& x, Eigen::Ref<MatrixXd> jac) {
    ALTRO_ASSERT(jac.rows() == jac.cols(), "Jacobian must be square.");
    ALTRO_UNUSED(x);
    jac.setIdentity();
  }
  static void Hessian(const VectorXdRef& x, const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
    ALTRO_ASSERT(hess.rows() == hess.cols(), "Hessian must be square.");
    ALTRO_ASSERT(x.size() == b.size(), "x and b must be the same size.");
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(b);
    hess.setZero();
  }
};

/**
 * @brief The space of all negative numbers, an alias for inequality constraints.
 *
 * Used to represent inequality constraints of the form:
 * \f[ h(x) \leq 0 \f]
 *
 * The negative orthant is a self-dual cone, and it's projection operator is
 * an element-wise `min(0, x)`.
 *
 */
class NegativeOrthant {
 public:
  NegativeOrthant() = delete;
  using DualCone = NegativeOrthant;

  static void Projection(const VectorXdRef& x, Eigen::Ref<VectorXd> x_proj) {
    ALTRO_ASSERT(x.size() == x_proj.size(), "x and x_proj must be the same size");
    for (int i = 0; i < x.size(); ++i) {
      x_proj(i) = std::min(0.0, x(i));
    }
  }
  static void Jacobian(const VectorXdRef& x, Eigen::Ref<MatrixXd> jac) {
    ALTRO_ASSERT(jac.rows() == jac.cols(), "Jacobian must be square.");
    for (int i = 0; i < x.size(); ++i) {
      jac(i, i) = x(i) > 0 ? 0 : 1;
    }
  }
  static void Hessian(const VectorXdRef& x, const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
    ALTRO_ASSERT(hess.rows() == hess.cols(), "Hessian must be square.");
    ALTRO_ASSERT(x.size() == b.size(), "x and b must be the same size.");
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(b);
    hess.setZero();
  }
};
using Inequality = NegativeOrthant;

/**
 * @brief An abstract constraint of the form:
 * \f[ g(x, u) \in K \f]
 *
 * where \f$ K \f$ is an arbitrary convex cone, specified by @tparam ConType.
 * This formulation supports generic equality and inequality constraints.
 *
 * # Interface
 * The user is expected to implement the folowing interface when defining a constraint:
 * - `int OutputDimension() const` - size of output (length of constraint).
 * - `void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<Eigen::VectorXd> out)`
 * - `void Jacobian(const VectorXdRef& x, const VectorXdRef& u, JacobianRef out)`
 *
 * Where we use the following Eigen type aliases:
 *    using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>
 *    using JacobianRef = Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
 * Eigen::RowMajor>>
 *
 * The Jacobian is stored row-major since Jacobians are naturally evaluated
 * row-wise. Storing the underlying data in row-major format allows the rows to
 * be processed individually in a cache-friendly way.
 *
 * The constraint is required to at least have continuous 1st order derivatives,
 * and these derivatives must be implemented by the user. No automatic or
 * approximation differentiation methods are provided, although the Jacobian
 * can be verified using a finite difference method using `CheckJacobian`. See
 * documentation for `FunctionBase` for more information.
 *
 * @tparam ConType The type of constraint (equality, inequality, conic, etc.)
 */
template <class ConType>
class Constraint : public FunctionBase {
 public:
  // These aren't used right now, but they need to be defined.
  int StateDimension() const override {
    ALTRO_ASSERT(false, "StateDimension hasn't been defined for this constraint.");
    return -1;
  }
  int ControlDimension() const override {
    ALTRO_ASSERT(false, "ControlDimension hasn't been defined for this constraint.");
    return -1;
  }

  // TODO(bjackson) [SW-14476] add 2nd order terms when implementing DDP
  bool HasHessian() const override { return false; }
};

template <class ConType>
using ConstraintPtr = std::shared_ptr<Constraint<ConType>>;

}  // namespace constraints
}  // namespace altro