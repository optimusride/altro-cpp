#pragma once
#include <fmt/format.h>
#include <fmt/ostream.h>

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
class ConstraintValues : public Constraint<ConType> {
  static constexpr int p = Eigen::Dynamic;
  static constexpr int n_m = AddSizes(n, m);

 public:
  static constexpr double kDefaultPenaltyScaling = 10.0;
  /**
   * @brief Construct a new Constraint Values object
   *
   * @param state_dim state dimension
   * @param control_dim  control dimension
   * @param con Pointer to a constraint. Assumes that the constraint function
   * can be evaluated with inputs that are consistent with state_dim and control_dim
   */
  ConstraintValues(const int state_dim, const int control_dim, ConstraintPtr<ConType> con)
      : n_(state_dim), m_(control_dim), con_(std::move(con)) {
    int output_dim = con_->OutputDimension();
    c_.setZero(output_dim);
    lambda_.setZero(output_dim);
    penalty_.setOnes(output_dim);
    jac_.setZero(output_dim, state_dim + control_dim);
    hess_.setZero(state_dim + control_dim, state_dim + control_dim);
    lambda_proj_.setZero(output_dim);
    c_proj_.setZero(output_dim);
    proj_jac_.setZero(output_dim, output_dim);
    jac_proj_.setZero(output_dim, state_dim + control_dim);
  }

  /***************************** Getters **************************************/
  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }

  VectorNd<p>& GetDuals() { return lambda_; }
  VectorNd<p>& GetPenalty() { return penalty_; }
  VectorNd<p>& GetConstraintValue() { return c_; }
	double GetPenaltyScaling() const { return penalty_scaling_; }

  /***************************** Setters **************************************/
  /**
   * @brief Set the same penalty for all constraints
   *
   * @param rho Penalty value. rho >= 0.
   */
  void SetPenalty(double rho) {
    ALTRO_ASSERT(rho >= 0, "Penalty must be positive.");
    penalty_.setConstant(rho);
  }

  void SetPenaltyScaling(double phi) { 
    ALTRO_ASSERT(phi >= 1, "Penalty must be greater than 1.");
    penalty_scaling_ = phi; 
  }

  /***************************** Methods **************************************/
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
   * penalty parameter, and \f$ \Pi_{K^*} \f$ is the projection operator for the dual cone \f$ K^*
   * \f$.
   *
   * @param x State vector
   * @param u Control vector
   * @return The augmented Lagrangian for the current knot point, evaluated at x and u.
   */
  double AugLag(const VectorXdRef& x, const VectorXdRef& u) {
    const double rho = penalty_(0);
    con_->Evaluate(x, u, c_);

    ConType::DualCone::Projection(lambda_ - rho * c_, lambda_proj_);
    double J = lambda_proj_.squaredNorm() - lambda_.squaredNorm();
    J = J / (2 * rho);
    return J;
  }

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
                      Eigen::Ref<VectorXd> du) {
    const double rho = penalty_(0);

    // TODO(bjackson): Avoid these redundant calls.
    con_->Evaluate(x, u, c_);
    con_->Jacobian(x, u, jac_);
    ConType::DualCone::Projection(lambda_ - rho * c_, lambda_proj_);
    ConType::DualCone::Jacobian(lambda_ - rho * c_, proj_jac_);
    const int output_dim = con_->OutputDimension();
    dx = -(proj_jac_ * jac_.topLeftCorner(output_dim, this->n_)).transpose() * lambda_proj_;
    du = -(proj_jac_ * jac_.topRightCorner(output_dim, this->m_)).transpose() * lambda_proj_;
  }
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
                     Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu, const bool full_newton) {
    const double rho = penalty_(0);

    // TODO(bjackson): Avoid these redundant calls.
    con_->Evaluate(x, u, c_);
    con_->Jacobian(x, u, jac_);
    ConType::DualCone::Projection(lambda_ - rho * c_, lambda_proj_);
    ConType::DualCone::Jacobian(lambda_ - rho * c_, proj_jac_);
    jac_proj_ = proj_jac_ * jac_;
    const int output_dim = con_->OutputDimension();
    dxdx = rho * jac_proj_.topLeftCorner(output_dim, this->n_).transpose()
           * jac_proj_.topLeftCorner(output_dim, this->n_);
    dxdu = rho * jac_proj_.topLeftCorner(output_dim, this->n_).transpose()
           * jac_proj_.topRightCorner(output_dim, this->m_);
    dudu = rho * jac_proj_.topRightCorner(output_dim, this->m_).transpose()
           * jac_proj_.topRightCorner(output_dim, this->m_);

    if (full_newton) {
      throw std::runtime_error("Second-order constraint terms are not yet supported.");
    }
  }

  /**
   * @brief Update the dual variables
   * 
   * Updates the dual variables using the current constraint and penalty values.
   * The resulting dual variables are projected back into the dual cone such that
   * they are always guaranteed to be feasible with respect to the dual cone.
   * 
   * The update is of the form:
   * \f[
   * \lambda^+ - \Pi_{K^*}(\lambda - \rho c)
   * \f]
   * 
   */
  void UpdateDuals() {
    ConType::DualCone::Projection(lambda_ - penalty_.asDiagonal() * c_, lambda_);
  }

  /**
   * @brief Update the penalty parameters
   * 
   * For now just does a naive uniform geometric increase.
   * 
   */
  void UpdatePenalties() {
    // TODO(bjackson): Look into more advanced methods for updating the penalty parameter
    penalty_ *= penalty_scaling_;
    const double rho = penalty_(0);
    (void) rho;
  }

  /**
   * @brief Calculate the maximum constraint violation
   * 
   * @tparam p The norm to use when calculating the violation (default = Infinity)
   * @return Maximum constraint violation 
   */
  template <int p = Eigen::Infinity>
  double MaxViolation() {
    ConType::Projection(c_, c_proj_);
    c_proj_ = c_ - c_proj_;
    return c_proj_.template lpNorm<p>();
  }

  /**
   * @brief Find the maximum penalty
   * 
   * @return The maximum penalty parameter 
   */
  double MaxPenalty() { return penalty_.maxCoeff(); }

  // Pass constraint interface to internal pointer
  static constexpr int NStates = n;
  static constexpr int NControls = m;
  int OutputDimension() const override { return con_->OutputDimension(); }
  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    con_->Evaluate(x, u, c);
  }
  void Jacobian(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<MatrixXd> jac) override {
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
  const int n_;  // state dimension
  const int m_;  // control dimension
  ConstraintPtr<ConType> con_;
  VectorNd<p> c_;            // constraint value
  VectorNd<p> lambda_;       // Langrange multiplier
  VectorNd<p> penalty_;      // penalty values
  MatrixNxMd<p, n_m> jac_;    // Jacobian
  MatrixNxMd<n_m, n_m> hess_;  // Hessian

  VectorNd<p> lambda_proj_;     // projected multiplier
  VectorNd<p> c_proj_;          // projected constraint value
  MatrixNxMd<p, p> proj_jac_;   // Jacobian of projection operation
  MatrixNxMd<p, n_m> jac_proj_;  // Jacobian through projection operation (jac_ * proj_jac_)

  double penalty_scaling_ = kDefaultPenaltyScaling;
};

}  // namespace constraints
}  // namespace altro