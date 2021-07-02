#pragma once

#include <memory>
#include <vector>

#include "altro/eigentypes.hpp"
#include "altro/problem/problem.hpp"
#include "altro/problem/dynamics.hpp"
#include "altro/problem/costfunction.hpp"

namespace altro {
namespace problem {

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
 public:
  /**
   * @brief Initialize a new Problem with N segments
   *
   * @param N number of trajectory segments (1 less than the number of knot
   * points)
   */
  explicit Problem(const int N)
      : N_(N), costfuns_(N + 1, nullptr), models_(N, nullptr) {}


  /**
   * @brief Set the initial state for the problem
   * 
   * @param x0 the initial state
   */
  void SetInitialState(const Eigen::Ref<const VectorXd>& x0) {
    initial_state_ = x0;
  }

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
   * @brief Set the cost function for an interval of knot points 
   * 
   * Set all knot points in the interval [k_start, k_stop) to the same cost 
   * function.
   * 
   * To set all knot points to have the same cost function, use
   * @code {.cpp}
   * SetCostFunction(costfun, 0, N+1) ;
   * @endcode
   * 
   * @param costfun Cost function object pointer
   * @param k_start Starting index (inclusive) (0 <= k_start < k_stop)
   * @param k_stop  Terminal index (exclusive) (k_start < k_stop <= N)
   */
  void SetCostFunction(std::shared_ptr<CostFunction> costfun, int k_start,
                       int k_stop);

  /**
   * @brief Set the dynamics model at time step k 
   * 
   * @param model Dynamics function object pointer
   * @param k time step (0 <= k < N)
   */
  void SetDynamics(std::shared_ptr<DiscreteDynamics> model, int k) {
    ALTRO_ASSERT((k >= 0) && (k < N_), "Invalid knot point index.");
    models_[k] = std::move(model);
  }

  /**
   * @brief Set the dynamics model for an interval of knot points 
   * 
   * Set all knot points in the interval [k_start, k_stop) to the same dynamics 
   * model. 
   * 
   * To set all knot points to have the same dynamics model. 
   * @code {.cpp}
   * SetDynamics(costfun, 0, N) ;
   * @endcode
   * 
   * @param costfun Cost function object pointer
   * @param k_start Starting index (inclusive) (0 <= k_start < k_stop)
   * @param k_stop  Terminal index (exclusive) (k_start < k_stop < N)
   */
  void SetDynamics(std::shared_ptr<DiscreteDynamics> model, int k_start,
                   int k_stop);

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
   * @param k Must be in range [0, N)
   * @return Shared pointer to the cost function object
   */
  std::shared_ptr<DiscreteDynamics> GetDynamics(int k) const {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    if (k == N_) {
      return nullptr;
    }
    return models_[k];
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
};

}  // namespace problem
}  // namespace altro