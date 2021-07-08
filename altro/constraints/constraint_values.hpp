#pragma once

#include "altro/common/state_control_sized.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/eigentypes.hpp"

namespace altro {
namespace constraints {

/**
 * @brief A constraint that also allocates memory for constraint values, Jacobians,
 * dual variables, etc.
 *
 * This class also provides methods for evaluating terms such as the augmented Lagrangian.
 *
 * @tparam n Compile-time state dimension
 * @tparam m Compile-time control dimension
 * @tparam ConType Type of constraint (equality, inequality, conic, etc.)
 */
template <int n, int m, class ConType>
class ConstraintValues : public StateControlSized<n, m>, Constraint<ConType> {
  static constexpr int p = Eigen::Dynamic;

 public:
  /**
   * @brief Construct a new Constraint Values object
   *
   * @param state_dim state dimension
   * @param control_dim  control dimension
   * @param con Pointer to a constraint. Assumes that the constraint function
   * can be evaluated with inputs that are consistent with state_dim and control_dim
   */
  ConstraintValues(const int state_dim, const int control_dim, ConstraintPtr<ConType> con)
      : StateControlSized<n, m>(state_dim, control_dim), con_(std::move(con)) {
    int output_dim = con_->OutputDimension();
    c_.setZero(output_dim);
    lambda_.setZero(output_dim);
    penalty_.setZero(output_dim);
    jac_.setZero(output_dim, state_dim + control_dim);
    hess_.setZero(state_dim + control_dim, state_dim + control_dim);
  }

  /**
   * @brief Evaluate the augmented Lagrangian
   * 
   * The augmented Lagrangian for an optimization problem of the form:
   * \f{aligned}{
   *   \text{minimize} &&& f(x) \\
   *   \text{subject to} &&& c(x) \in K \\
   * \f}
   * 
   * is defined to be 
   * \f[ 
   * f(x) + \frac{1}{2 \rho} (||\Pi_{K^*}(\lambda - \rho c(x))||_2^2 - ||\lambda||_2^2)
   * \f]
   * where \f$ \lambda \f$ are the Lagrange multipliers (dual variables), \f$ \rho \f$ is a scalar
   * penalty parameter, and \f$ \Pi_{K^*} \f$ is the projection operator for the dual cone \f$ K^* \f$.
   * 
   * @param x State vector
   * @param u Control vector
   * @return The augmented Lagrangian for the current knot point, evaluated at x and u.
   */
  double AugLag(const VectorXdRef& x, const VectorXdRef& u) const;

  /**
   * @brief The gradient of the Augmented Lagrangian
   * 
   * Uses the Jacobian of the projection operator for the dual cone.
   * 
   * @param[in] x State vector
   * @param[in] u Control vector
   * @param[out] dx Gradient with respect to the states.
   * @param[out] du Gradient with respect to the controls.
   */
  void AugLagGradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                      Eigen::Ref<VectorXd> du) const;
  /**
   * @brief The Hessian of the Augmented Lagrangian
   * 
   * Uses the Jacobian of the Jacobian-transpose-vector-proeduct of the projection operator for the 
   * dual cone and the constraint.
   * 
   * @param[in] x State vector
   * @param[in] u Control vector
   * @param[out] dxdx Hessian with respect to the states.
   * @param[out] dxdu Hessian cross-term with respect to the state and controls.
   * @param[out] dudu Hessian with respect to the controls. 
   */
  void AugLagHessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                     Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu,
                     const bool full_newton) const;

  // Pass constraint interface to internal pointer
  int OutputDimension() const override { return con_->OutputDimension(); }
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) const override {
    con_->Evaluate(x, u, c);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<MatrixXd> jac) const override {
    con_->Jacobian(x, u, jac);
  }

  /**
   * @brief Evaluate the constraint and it's derivatives
   * 
   * Stores the result internally.
   * 
   * @param[in] x State vector
   * @param[in] u Control vector
   */
  void CalcExpansion(const VectorXdRef& x, const VectorXdRef& u) {
    con_->Evaluate(x, u, c_);
    con_->Jacobian(x, u, jac_);
  }

 private:
  ConstraintPtr<ConType> con_;
  VectorNd<p> c_;                 // constraint value
  VectorNd<p> lambda_;            // langrange multiplier
  VectorNd<p> penalty_;           // penalty values
  MatrixNxMd<p, n + m> jac_;       // Jacobian
  MatrixNxMd<n + m, n + m> hess_;  // Hessian
};

/**
 * @brief Holds all of the constraints at a single knot point
 *
 * @tparam n Compile-time state dimension
 * @tparam m Compile-time control dimension
 */
template <int n, int m>
class ConstraintSet {
 public:
  
  /**
   * @brief Evaluates the sum of the augmented Lagrangian for all of the constraints
   * at the current knot point.
   * 
   * @param[in] x State vector
   * @param[in] u Control vector
   * @return double 
   */
  double AugLag(const VectorXdRef& x, const VectorXdRef& u) const {
    double J = 0.0;
    for (const auto conval : eq_) {
      J += conval->AugLag(x, u);
    }
    for (const auto conval : ineq_) {
      J += conval->AugLag(x, u);
    }
    // TODO(bjackson) [SW-14871]: Add Second-order cone constraints
    // for (const auto conval : soc_) {
    //   J += conval->AugLag(x, u);
    // }
    return J;
  }

  void AugLagGradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                      Eigen::Ref<VectorXd> du) const {
    for (const auto conval : eq_) {
      conval->AugLagGradient(x, u, dx, du);
    }
    for (const auto conval : ineq_) {
      conval->AugLagGradient(x, u, dx, du);
    }
    // TODO(bjackson) [SW-14871]: Add Second-order cone constraints
    // for (const auto conval : soc_) {
    //   conval->AugLagGradient(x, u, dx, du);
    // }
  }

  void AugLagHessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                     Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu,
                     const bool full_newton = false) const {
    for (const auto conval : eq_) {
      conval->AugLagHessian(x, u, dxdx, dxdu, dudu, full_newton);
    }
    for (const auto conval : ineq_) {
      conval->AugLagHessian(x, u, dxdx, dxdu, dudu, full_newton);
    }
    // TODO(bjackson) [SW-14871]: Add Second-order cone constraints
    // for (const auto conval : soc_) {
    //   conval->AugLagHessian(x, u, dxdx, dxdu, dudu, full_newton);
    // }
  }

  template <class ConType>
  void AddConstraint(ConstraintPtr<ConType> con);

 private:
  std::vector<std::shared_ptr<ConstraintValues<n, m, Equality>>> eq_;
  std::vector<std::shared_ptr<ConstraintValues<n, m, NegativeOrthant>>> ineq_;
};

}  // namespace constraints
}  // namespace altro