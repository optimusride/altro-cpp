#pragma once

#include <vector>

#include "eigentypes.hpp"
#include "problem/problem.hpp"
#include "problem/dynamics.hpp"
#include "problem/costfunction.hpp"

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
class Problem
{
 public:
  /**
   * @brief Construct a new Problem object, with different costs at each time step
   * 
   * @param dynamics a vector of N-1 dynamics objects for evaluating the discrete dynamics
   * @param costs a vector of N cost function objects for evaluating the cost function
   * @param h a vector of N-1 steps in the independent variable (e.g. time) 
   * @param x0 initial state
   */
  Problem(const std::vector<DiscreteDynamics>& dynamics, const std::vector<CostFunction>& costs, 
          const std::vector<float> h, VectorXd x0);

  /**
   * @brief Construct a new Problem object, with uniform dynamics, costs, and 
   * discretization steps
   * 
   * @param dynamics an object for defining the discrete dynamics
   * @param cost an object for evaluating the cost function
   * @param N number of knot points
   * @param h discretization step
   * @param x0 initial state
   */
  Problem(const DiscreteDynamics& dynamics, const CostFunction& cost, size_t N, 
          float h, const VectorXd& x0);

  CostFunction& GetCostFunction(int k);
  DiscreteDynamics& GetDynamics(int k);
  float GetStep(int k);
};

} // namespace problem 
} // namespace altro