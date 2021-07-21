#pragma once

#include "altro/augmented_lagrangian/al_cost.hpp"
#include "altro/problem/problem.hpp"

namespace altro {
namespace augmented_lagrangian {

/**
 * @brief Build the augmented Lagrangian trajectory optimization problem.
 *
 * Takes a constrained trajectory optimization problem and creates an unconstrained
 * trajectory optimization problem by moving the constraints into the cost function
 * using an augmented Lagrangian cost. Each cost function is an ALCost<n, m>.
 *
 * @tparam n Compile-time state dimension.
 * @tparam m Compile-time control dimension.
 * @param[in] prob The original, potentially constrained, optimization problem.
 * @param[out] costs Optional container that will be populated with the ALCost
 * types that are assigned to the problem. Useful since the problem only stores
 * generic CostFunction pointers.
 *
 * @return problem::Problem A new unconstrained trajectory optimization problem
 * with an augmented Lagrangian cost function containing the constraints of the
 * original problem.
 */
template <int n, int m>
problem::Problem BuildAugLagProblem(const problem::Problem& prob,
                                    std::vector<std::shared_ptr<ALCost<n, m>>>* costs = nullptr) {
  const int N = prob.NumSegments();
  problem::Problem prob_al(N);

  // Copy initial state and dynamics
  prob_al.SetInitialState(prob.GetInitialState());
  for (int k = 0; k < N; ++k) {
    prob_al.SetDynamics(prob.GetDynamics(k), k);
  }

  // Create an augmented Lagrangian cost function combining the original cost
  // function and the constraints
  for (int k = 0; k <= N; ++k) {
    std::shared_ptr<ALCost<n, m>> alcost = std::make_shared<ALCost<n, m>>(prob, k);
    if (costs) {
      costs->emplace_back(alcost);
    }
    prob_al.SetCostFunction(std::move(alcost), k);
  }
  return prob_al;
}

}  // namespace augmented_lagrangian
}  // namespace altro