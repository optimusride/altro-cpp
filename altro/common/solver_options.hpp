#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "altro/utils/utils.hpp"
#include "altro/common/solver_stats.hpp"

namespace altro {

/**
 * @brief Options for augmented Lagrangian and iLQR solvers
 * 
 */
struct SolverOptions {
  // clang-format off
  // NOLINT comments added to supress clang-tidy [readibility-magic-numbers] check
  int max_iterations_total = 300;         // NOLINT Maximum number of total iterative LQR iterations
  int max_iterations_outer = 30;          // NOLINT Maximum augmented Lagrangian iterations
  int max_iterations_inner = 100;         // NOLINT Max iLQR iterations in a single solve
  double cost_tolerance = 1e-4;           // NOLINT Threshold for cost decrease 
  double gradient_tolerance = 1e-2;       // NOLINT Threshold for infinity-norm of the approximate gradient

  double bp_reg_increase_factor = 1.6;    // NOLINT Multiplicative factor for increasing the regularization
  bool bp_reg_enable = true;              // NOLINT Enable regularization in the backward pass
  double bp_reg_initial = 0.0;            // NOLINT Initial regularization
  double bp_reg_max = 1e8;                // NOLINT Maximum regularization
  double bp_reg_min = 1e-8;               // NOLINT Minimum regularization
  // double bp_reg_forwardpass = 10.0;     
  int bp_reg_fail_threshold = 100;        // NOLINT How many time the backward pass can fail before throwing an error
  bool check_forwardpass_bounds = true;   // NOLINT Whether to check if the rollouts stay within the specified bounds
  double state_max = 1e8;                 // NOLINT Maximum state value (abs)
  double control_max = 1e8;               // NOLINT Maximum control value (abs)

  int line_search_max_iterations = 20;    // NOLINT Maximum number of line search iterations before increasing regularization
  double line_search_lower_bound = 1e-8;  // NOLINT Sufficient improvement condition
  double line_search_upper_bound = 10.0;  // NOLINT Can't make too much more improvement than expected
  double line_search_decrease_factor = 2; // NOLINT How much the line search step size is decreased each iteration

  double constraint_tolerance = 1e-4;     // NOLINT Maximum constraint violation theshold
  double maximum_penalty = 1e8;           // NOLINT Maximum penalty parameter allowed
  LogLevel verbose = LogLevel::kSilent;   // Output verbosity level
  // clang-format on
};

}  // namespace altro