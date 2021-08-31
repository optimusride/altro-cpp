// Copyright [2021] Optimus Ride Inc.

#pragma once

#include <memory>

#include "altro/common/state_control_sized.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace augmented_lagrangian {

/**
 * @brief An augmented Lagrangian cost function, which adds the linear and
 * quadratic penalty costs to an existing cost function.
 *
 * It defines a cost of the following form
 * \f[
 * f(x,u) + \frac{1}{2 \rho} (|| \Pi_{K^*}(\lambda - \rho c(x, u)) ||_2^2 - || \lambda ||_2^2)
 * \f]
 * where \f$ \lambda \in \mathbb{R}^p \f$ are the Lagrange multipliers,
 * \f$ \rho > 0 \in \mathbb{R} \f$ is a penalty parameter, and \f$ \Pi_{K^*}(\cdot) \f$ is
 * the projection operator for the dual cone \f$ K^* \f$. For equality constraints
 * this is simply the identity mapping.
 *
 * The constraints are stored as ConstraintValues, which internally store the
 * dual variables and penalty parameters.
 *
 * @tparam n Compile-time state dimension.
 * @tparam m Compile-time control dimension.
 */
template <int n, int m>
class ALCost : public problem::CostFunction {
  template <class ConType>
  using ConstraintValueVec =
      std::vector<std::shared_ptr<constraints::ConstraintValues<n, m, ConType>>>;

 public:
  ALCost(const int state_dim, const int control_dim) : n_(state_dim), m_(control_dim) { Init(); }

  /**
   * @brief Construct a new ALCost object from a Problem
   *
   * Generates an ALCost by combining the cost function and constraints at the specified knot point
   * index, excluding the dynamics constraints.
   *
   * @param prob Description of the constrained trajectory optimization problem
   * @param k Index of the knot point. 0 <= k <= prob.NumSegments()
   */
  ALCost(const problem::Problem& prob, const int k)
      : n_(prob.GetDynamics(k)->StateDimension()), m_(prob.GetDynamics(k)->ControlDimension()) {
    SetCostFunction(prob.GetCostFunction(k));
    SetEqualityConstraints(prob.GetEqualityConstraints().at(k).begin(),
                           prob.GetEqualityConstraints().at(k).end());
    SetInequalityConstraints(prob.GetInequalityConstraints().at(k).begin(),
                             prob.GetInequalityConstraints().at(k).end());
    Init();
  }

  /***************************** Getters **************************************/

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }
  static constexpr int NStates = n;
  static constexpr int NControls = m;

  ConstraintValueVec<constraints::Equality>& GetEqualityConstraints() { return eq_; }
  ConstraintValueVec<constraints::Inequality>& GetInequalityConstraints() { return ineq_; }

  /**
   * @brief Calculate the length of the constraint vector associated with the cost function.
   *
   * @return int Length of the constraint vector, including all constraint types (equalities,
   * inequalities, conic, etc.)
   */
  int NumConstraints() {
    int p = 0;
    for (size_t i = 0; i < eq_.size(); ++i) {
      p += eq_[i]->OutputDimension();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      p += ineq_[i]->OutputDimension();
    }
    return p;
  }

  std::shared_ptr<problem::CostFunction> GetCostFunction() { return costfun_; }

  /**
   * @brief Append the constraint info for all the constraints in the cost.
   * 
   * @param coninfo A vector of constraint info. The constraint info for the
   * current cost is appended on to the end of the vector.
   */
  void GetConstraintInfo(std::vector<constraints::ConstraintInfo>* coninfo) {
    for (const auto& conval : eq_) {
      coninfo->emplace_back(conval->GetConstraintInfo());
    }
    for (const auto& conval : ineq_) {
      coninfo->emplace_back(conval->GetConstraintInfo());
    }
  }

  /***************************** Setters **************************************/

  /**
   * @brief Assign the nominal cost function
   *
   * @param costfun Pointer to an instantiation of the CostFunction interface.
   */
  void SetCostFunction(const std::shared_ptr<problem::CostFunction>& costfun) {
    ALTRO_ASSERT(costfun != nullptr, "Cost function cannot be a nullptr.");
    costfun_ = costfun;
  }

  // TODO(bjackson): Find a way to unify these by templating over the ConType.
  // Non-trivial since it requires specialization of a method within a generic template class.

  /**
   * @brief Assign the equality constraints
   *
   * Accepts an arbitrary iterator pair. When de-referenced, the iterator must return a
   * std::shared_ptr<Constraint<Equality>>.
   *
   * Creates a ConstraintValues type for each constraint, where the constraint values, Jacobian,
   * dual variables, and penalty parameters associated with the constraint are stored.
   *
   * @tparam Iterator An iterator over pointers to equality constraints. Must be copyable.
   * @param begin Starting iterator
   * @param end Terminal iterator.
   */
  template <class Iterator>
  void SetEqualityConstraints(const Iterator& begin, const Iterator& end) {
    eq_.clear();
    CopyToConstraintValues<Iterator, constraints::Equality>(begin, end, &eq_);
    eq_tmp_ = VectorXd::Zero(eq_.size());
  }

  /**
   * @brief Assign the inequality constraints
   *
   * Accepts an arbitrary iterator pair. When de-referenced, the iterator must return a
   * std::shared_ptr<Constraint<Inequality>>.
   *
   * Creates a ConstraintValues type for each constraint, where the constraint values, Jacobian,
   * dual variables, and penalty parameters associated with the constraint are stored.
   *
   * @tparam Iterator An iterator over pointers to inequality constraints. Must be copyable.
   * @param begin Starting iterator
   * @param end Terminal iterator.
   */
  template <class Iterator>
  void SetInequalityConstraints(const Iterator& begin, const Iterator& end) {
    ineq_.clear();
    CopyToConstraintValues<Iterator, constraints::Inequality>(begin, end, &ineq_);
    ineq_tmp_ = VectorXd::Zero(ineq_.size());
  }

  /**
   * @brief Set the Penalty parameter for a specific constraint.
   *
   * Applies to all of the elements of that constraint.
   *
   * @tparam ConType Constraint type
   * @param rho Penalty parameter (rho > 0).
   * @param i Constraint index. 0 <= i <= NumConstraintFunctions<ConType>().
   */
  template <class ConType>
  void SetPenalty(const double rho, const int i) {
    ALTRO_ASSERT(0 <= i && i < NumConstraintFunctions<ConType>(),
                 fmt::format("Invalid constraint index. Got {}, expected to be in range [{},{})", i,
                             0, NumConstraintFunctions<ConType>()));
    if (std::is_same<ConType, constraints::Equality>::value) {
      eq_.at(i)->SetPenalty(rho);
    } else if (std::is_same<ConType, constraints::Inequality>::value) {
      ineq_.at(i)->SetPenalty(rho);
    }
  }

  /**
   * @brief Set the same penalty parameter for all constraints of the same type
   *
   * @tparam ConType Constraint type
   * @param rho Penalty parameter (rho > 0);
   */
  template <class ConType>
  void SetPenalty(const double rho) {
    int num_cons = NumConstraintFunctions<ConType>();
    for (int i = 0; i < num_cons; ++i) {
      SetPenalty<ConType>(rho, i);
    }
  }

  /**
   * @brief Set the Penalty scaling parameter.
   *
   * The penalty scaling is the multiplicative factor by which the penalties are updated.
   *
   * @tparam ConType Constraint type.
   * @param phi Penalty scaling parameter (phi > 1).
   * @param i
   */
  template <class ConType>
  void SetPenaltyScaling(const double phi, const int i) {
    ALTRO_ASSERT(0 <= i && i < NumConstraintFunctions<ConType>(),
                 fmt::format("Invalid constraint index. Got {}, expected to be in range [{},{})", i,
                             0, NumConstraintFunctions<ConType>()));
    if (std::is_same<ConType, constraints::Equality>::value) {
      eq_.at(i)->SetPenaltyScaling(phi);
    } else if (std::is_same<ConType, constraints::Inequality>::value) {
      ineq_.at(i)->SetPenaltyScaling(phi);
    }
  }

  /**
   * @brief Set the same penalty scaling parameter for all constraints of the same
   * type
   *
   * @tparam ConType Constraint type
   * @param rho Penalty scaling parameter (phi > 1).
   */
  template <class ConType>
  void SetPenaltyScaling(const double phi) {
    int num_cons = NumConstraintFunctions<ConType>();
    for (int i = 0; i < num_cons; ++i) {
      SetPenaltyScaling<ConType>(phi, i);
    }
  }

  /**
   * @brief Get the number of constraint functions of a particular type
   *
   * NumConstraintFunctions() <=  NumConstraints() (equality only when the
   * output dimension of each constraint function is 1).
   *
   * @tparam ConType Type of constraint function (e.g. Equality, Inequality, etc.)
   * @return int Number of constraint functions.
   */
  template <class ConType>
  int NumConstraintFunctions() {
    int size = 0.0;
    if (std::is_same<ConType, constraints::Equality>::value) {
      size = eq_.size();
    } else if (std::is_same<ConType, constraints::Inequality>::value) {
      size = ineq_.size();
    }
    return size;
  }

  /***************************** Methods **************************************/

  /**
   * @brief Evaluate the augmented Lagrangian cost
   *
   * @param x State vector
   * @param u Control vector
   * @return double Nominal cost plus the extra terms from the constraint penalties.
   *
   * @pre The cost function must be set prior to calling this function.
   */
  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override {
    ALTRO_ASSERT(costfun_ != nullptr, "Cost function must be set before evaluating.");
    double J = costfun_->Evaluate(x, u);
    for (size_t i = 0; i < eq_.size(); ++i) {
      J += eq_[i]->AugLag(x, u);
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      J += ineq_[i]->AugLag(x, u);
    }
    return J;
  }

  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                Eigen::Ref<VectorXd> du) override {
    ALTRO_ASSERT(costfun_ != nullptr, "Cost function must be set before evaluating.");
    costfun_->Gradient(x, u, dx, du);
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->AugLagGradient(x, u, dx_tmp_, du_tmp_);
      dx += dx_tmp_;
      du += du_tmp_;
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->AugLagGradient(x, u, dx_tmp_, du_tmp_);
      dx += dx_tmp_;
      du += du_tmp_;
    }
  }

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
               Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) override {
    ALTRO_ASSERT(costfun_ != nullptr, "Cost function must be set before evaluating.");
    costfun_->Hessian(x, u, dxdx, dxdu, dudu);
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->AugLagHessian(x, u, dxdx_tmp_, dxdu_tmp_, dudu_tmp_, full_newton_);
      dxdx += dxdx_tmp_;
      dxdu += dxdu_tmp_;
      dudu += dudu_tmp_;
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->AugLagHessian(x, u, dxdx_tmp_, dxdu_tmp_, dudu_tmp_, full_newton_);
      dxdx += dxdx_tmp_;
      dxdu += dxdu_tmp_;
      dudu += dudu_tmp_;
    }
  }

  /**
   * @brief Apply the dual update to all of the constraints
   *
   */
  void UpdateDuals() {
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->UpdateDuals();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->UpdateDuals();
    }
  }

  /**
   * @brief Apply the penalty update to all of the constraints
   *
   */
  void UpdatePenalties() {
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_[i]->UpdatePenalties();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_[i]->UpdatePenalties();
    }
  }

  /**
   * @brief Find the maximum constraint violation for the current knot
   * point
   *
   * @tparam p Norm to use when calculating the violation (default is Infinity)
   * @return Maximum constraint violation
   */
  template <int p = Eigen::Infinity>
  double MaxViolation() {
    for (size_t i = 0; i < eq_.size(); ++i) {
      eq_tmp_(i) = eq_[i]->template MaxViolation<p>();
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      ineq_tmp_(i) = ineq_[i]->template MaxViolation<p>();
    }
    Eigen::Vector2d tmp(eq_tmp_.template lpNorm<p>(), ineq_tmp_.template lpNorm<p>());
    return tmp.template lpNorm<p>();
  }

  /**
   * @brief Find the maximum penalty parameter being used by any of the
   * constraints at the current knot point.
   *
   * @return Maximum penalty parameter across all constraints.
   */
  double MaxPenalty() {
    double max_penalty = 0.0;
    for (size_t i = 0; i < eq_.size(); ++i) {
      max_penalty = std::max(max_penalty, eq_[i]->MaxPenalty());
    }
    for (size_t i = 0; i < ineq_.size(); ++i) {
      max_penalty = std::max(max_penalty, ineq_[i]->MaxPenalty());
    }
    return max_penalty;
  }

  void ResetDualVariables() {
    for (auto& con: eq_) {
      con->ResetDualVariables();
    }
    for (auto& con: ineq_) {
      con->ResetDualVariables();
    }
  }

 private:
  /**
   * @brief Allocates a new ConstraintValue for an arbitrary constraint, storing
   * the pointer in appropriately-typed vector.
   *
   *
   * @tparam Iterator An arbitrary iterator over pointers to constraints. Must be copyable.
   * @tparam ConType Type of contraint (Equality, Inequality, etc.)
   * @param begin Beginning iterator.
   * @param end Terminal iterator.
   * @param convals Container for storing the new ContraintValue.
   */
  template <class Iterator, class ConType>
  void CopyToConstraintValues(
      const Iterator& begin, const Iterator& end,
      std::vector<std::shared_ptr<constraints::ConstraintValues<n, m, ConType>>>* convals) {
    ALTRO_ASSERT(convals != nullptr, "Must provide a pointer to a valid collection.");
    for (Iterator it = begin; it != end; ++it) {
      convals->emplace_back(
          std::make_shared<constraints::ConstraintValues<n, m, ConType>>(this->n_, this->m_, *it));
    }
  }

  /**
   * @brief Allocate the temporary storage arrays.
   *
   */
  void Init() {
    dx_tmp_.setZero(this->n_);
    du_tmp_.setZero(this->m_);
    dxdx_tmp_.setZero(this->n_, this->n_);
    dxdu_tmp_.setZero(this->n_, this->m_);
    dudu_tmp_.setZero(this->m_, this->m_);
  }

  std::shared_ptr<problem::CostFunction> costfun_;

  // Constraints
  ConstraintValueVec<constraints::Equality> eq_;
  ConstraintValueVec<constraints::Inequality> ineq_;

  const int n_;
  const int m_;

  // Flag for using full/Gauss newton
  // TODO(bjackson): Add an option to change this, and use it in the expansion.
  bool full_newton_ = false;

  VectorXd eq_tmp_;
  VectorXd ineq_tmp_;

  // Arrays for collecting the cost expansion before adding
  // must be mutable since the CostFunction interface requires a const method
  VectorNd<n> dx_tmp_;
  VectorNd<m> du_tmp_;
  MatrixNxMd<n, n> dxdx_tmp_;
  MatrixNxMd<n, m> dxdu_tmp_;
  MatrixNxMd<m, m> dudu_tmp_;
};

}  // namespace augmented_lagrangian
}  // namespace altro