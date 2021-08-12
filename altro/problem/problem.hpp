#pragma once

#include <memory>
#include <vector>
#include <fmt/format.h>

#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/problem/dynamics.hpp"

namespace altro {
namespace problem {

/**
 * @brief Dummy discrete dynamics that don't do anything.
 * 
 * These dynamics are used at the last time step to provide the state dimension
 * to the downstream processes.
 * 
 */
class IdentityDynamics : public DiscreteDynamics {
 public:
  using DiscreteDynamics::Evaluate;
  explicit IdentityDynamics(int n, int m) : n_(n), m_(m) {
    ALTRO_ASSERT(n > 0, "State dimension must be greater than zero.");
    ALTRO_ASSERT(m > 0, "Control dimension must be greater than zero.");
  }

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& /*u*/, const float /*t*/,
                       const float /*h*/, Eigen::Ref<VectorXd> xnext) override {
    xnext = x;
  }
  void Jacobian(const VectorXdRef& /*x*/, const VectorXdRef& /*u*/, const float /*t*/,
                const float /*h*/, Eigen::Ref<MatrixXd> jac) override {
    jac.setIdentity();
  }
  void Hessian(const VectorXdRef& /*x*/, const VectorXdRef& /*u*/, const float /*t*/,
               const float /*h*/, const VectorXdRef& /*b*/,
               Eigen::Ref<MatrixXd> hess) override {
    hess.setZero();
  }
  bool HasHessian() const override { return true; }

 private:
  int n_;
  int m_;
};

/**
 * @brief Describes and evaluates the trajectory optimization problem
 *
 * Describes generic trajectory optimization problems of the following form:
 * minimize    sum( J(X[k], U[k]), k = 0:N )
 *   X,U
 * subject to f_k(X[k], U[k], X[k+1], U[k+1]) = 0, k = 0:N-1
 *            g_k(X[k], U[k]) = 0,                 k = 0:N
 *            h_ki(X[k], U[k]) in cone K_i,        k = 0:N, i = 0...
 */
class Problem {
  template <class ConType>
  using ConstraintSet = std::vector<constraints::ConstraintPtr<ConType>>;

 public:
  /**
   * @brief Initialize a new Problem with N segments
   *
   * @param N number of trajectory segments (1 less than the number of knot
   * points)
   */
  explicit Problem(const int N)
      : N_(N), costfuns_(N + 1, nullptr), models_(N + 1, nullptr), eq_(N + 1), ineq_(N + 1) {}

  /**
   * @brief Set the initial state for the problem
   *
   * @param x0 the initial state
   */
  void SetInitialState(const VectorXdRef& x0) { initial_state_ = x0; }

  /**
   * @brief Set the cost function at knot point k
   *
   * @param costfun Cost function object pointer
   * @param k knot point index (0 <= k <= N)
   */
  void SetCostFunction(std::shared_ptr<CostFunction> costfun, int k) {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    costfuns_[k] = std::move(costfun);
  }

  /**
   * @brief Set the cost function for an interval of consecutive knot points
   * 
   * Generally, each element of the input vector should be unique to ensure 
   * there are no race conditions when parallelizing over knot points.
   * 
   * @tparam CostFun A class derived from CostFunction.
   * @param costfuns A vector of cost function pointers. These pointers will by
   * copied into the problem and solver directly. It is the user's responsibility
   * to make sure that this operation does not result in race conditions when 
   * parallelized. To make sure this doesn't happen, the user should generally 
   * create a unique copy of the cost function for each knot point.
   * @param k_start Starting index (inclusive). All the pointers will be copied
   * starting from this knot point. Defaults to the start of the trajectory.
   */
  template <class CostFun>
  void SetCostFunction(const std::vector<std::shared_ptr<CostFun>>& costfuns, int k_start = 0) {
    for (size_t i = 0; i < costfuns.size(); ++i) {
      int k = i + k_start;
      SetCostFunction(costfuns[i], k);
    }
  }

  /**
   * @brief Set the dynamics model at time step k
   *
   * @param model Dynamics function object pointer
   * @param k time step (0 <= k < N)
   */
  void SetDynamics(std::shared_ptr<DiscreteDynamics> model, int k) {
    ALTRO_ASSERT(model != nullptr, "Cannot pass a nullptr for the dynamics.");
    ALTRO_ASSERT((k >= 0) && (k < N_), "Invalid knot point index.");

    // Create a dummy dynamics model at the last time step to provide the state
    // and control dimension
    if (k == N_ - 1) {
      models_.at(N_) =
          std::make_shared<IdentityDynamics>(model->StateDimension(), model->ControlDimension());
    }
    models_[k] = std::move(model);
  }

  /**
   * @brief Set the dynamics functions for an interval of consecutive knot points.
   * 
   * Generally, each element of the input vector should be unique to ensure 
   * there are no race conditions when parallelizing over knot points.
   * 
   * @tparam Dynamics A class derived from `problem::DiscreteDynamics`.
   * @param models A vector of dynamics function pointers. These pointers will by
   * copied into the problem and solver directly. It is the user's responsibility
   * to make sure that this operation does not result in race conditions when 
   * parallelized. To make sure this doesn't happen, the user should generally 
   * create a unique copy of the dynamics for each knot point. This is 
   * critically important for `DiscretizedModel`s, since these allocate temporary
   * storage for evaluating the numerical integration that cannot be used in a
   * thread-safe way without creating a new model for each knot point.
   * @param k_start Starting index (inclusive). All the pointers will be copied
   * starting from this knot point. Defaults to the start of the trajectory.
   */
  template <class Dynamics>
  void SetDynamics(const std::vector<std::shared_ptr<Dynamics>>& models, int k_start = 0) {
    for (size_t i = 0; i < models.size(); ++i) {
      int k = i + k_start;
      SetDynamics(models[i], k);
    }
  }

  template <class ConType>
  void SetConstraint(std::shared_ptr<constraints::Constraint<ConType>> con, int k);

  /**
   * @brief Count the length of the constraint vector at knotpoint k
   *
   * Note that this is the sum of the output dimensions for each constraint
   * function.
   *
   * @param k Knot point index 0 <= k <= N
   * @return Length of constraint vector at the knot point
   */
  int NumConstraints(const int k) {
    ALTRO_ASSERT(0 <= k && k <= N_, "k outside valid knot point indices.");
    int cnt = 0;
    for (const constraints::ConstraintPtr<constraints::Equality>& con : eq_.at(k)) {
      cnt += con->OutputDimension();
    }
    for (const constraints::ConstraintPtr<constraints::Inequality>& con : ineq_.at(k)) {
      cnt += con->OutputDimension();
    }
    return cnt;
  }

  /**
   * @brief Count the length of the constraint vector for the entire problem
   *
   * @return Total number of constraints
   */
  int NumConstraints() {
    int cnt = 0;
    for (int k = 0; k <= N_; ++k) {
      cnt += NumConstraints(k);
    }
    return cnt;
  }

  /**
   * @brief Get the initial state
   *
   * @return reference to the initial state vector
   */
  const VectorXd& GetInitialState() const { return initial_state_; }

  /**
   * @brief Get the Cost Function object at time step k
   *
   * @param k Must be in range [0, N]
   * @return Shared pointer to the cost function object
   */
  std::shared_ptr<CostFunction> GetCostFunction(int k) const {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    return costfuns_[k];
  }

  /**
   * @brief Get the Dynamics model object at time step k
   *
   * If the last knot point is requested, a null pointer will be returned.
   * Otherwise, a bad knot point index will result in an assertion failure.
   *
   * If the dynamics at the knot point are not defined, this function will result in an assertion
   * failure.
   *
   * @param k Must be in range [0, N)
   * @return Shared pointer to the cost function object, or nullptr if the dynamics at the last time
   * step are requested.
   *
   */
  std::shared_ptr<DiscreteDynamics> GetDynamics(int k) const {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    ALTRO_ASSERT(models_[k] != nullptr, "Dynamics have not been defined at this knot point.");
    return models_[k];
  }

  const std::vector<ConstraintSet<constraints::Equality>>& GetEqualityConstraints() const {
    return eq_;
  }
  const std::vector<ConstraintSet<constraints::Inequality>>& GetInequalityConstraints() const {
    return ineq_;
  }

  int NumSegments() const { return N_; }

  /**
   * @brief Check if the problem is fully defined
   *
   * A problem is fully defined if all the cost and dynamics model functions
   * pointers are not null pointers, and if the initial state and state
   * dimension at the first time step are consistent.
   *
   * @param verbose
   * @return true
   * @return false
   */
  bool IsFullyDefined(bool verbose = false) const;

 private:
  int N_;                   // number of segments (# of knotpoints - 1)
  VectorXd initial_state_;  // initial state
  std::vector<std::shared_ptr<CostFunction>> costfuns_;
  std::vector<std::shared_ptr<DiscreteDynamics>> models_;

  std::vector<ConstraintSet<constraints::Equality>> eq_;
  std::vector<ConstraintSet<constraints::Inequality>> ineq_;
};

}  // namespace problem
}  // namespace altro