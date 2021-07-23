#include "altro/problem/problem.hpp"

#include <iostream>

#include "altro/utils/assert.hpp"

namespace altro {
namespace problem {

void Problem::SetCostFunction(std::shared_ptr<CostFunction> costfun, int k_start, int k_stop) {
  ALTRO_ASSERT(k_start >= 0, "Invalid starting knot point index.");
  ALTRO_ASSERT(k_start < k_stop, "Starting index must be less than terminal index.");
  ALTRO_ASSERT(k_stop <= N_ + 1, "Invalid terminal knot point index.");
  for (int k = k_start; k < k_stop; ++k) {
    SetCostFunction(costfun, k);
  }
}

void Problem::SetDynamics(std::shared_ptr<DiscreteDynamics> model, int k_start, int k_stop) {
  ALTRO_ASSERT(k_start >= 0, "Invalid starting knot point index.");
  ALTRO_ASSERT(k_start < k_stop, "Starting index must be less than terminal index.");
  ALTRO_ASSERT(k_stop <= N_, "Invalid terminal knot point index.");
  for (int k = k_start; k < k_stop; ++k) {
    SetDynamics(model, k);
  }
}

bool Problem::IsFullyDefined(const bool verbose) const {
  bool valid = true;
  for (int k = 0; k <= N_; ++k) {
    bool has_costfun = (costfuns_[k] != nullptr);
    bool has_dynamics = (k < N_) ? (models_[k] != nullptr) : true;
    bool valid_k = (has_costfun && has_dynamics);

    bool has_initial_state = true;
    bool good_state_dim = true;
    if (k == 0) {
      has_initial_state = initial_state_.size() > 0;
      good_state_dim = has_initial_state && (models_[k]->StateDimension() == initial_state_.size());
      valid_k = valid_k && has_initial_state && good_state_dim;
    }
    valid = valid && valid_k;

    if (verbose) {
      std::cout << "Index " << k << ": ";
      std::cout << (valid_k ? "PASS " : "FAIL ");

      if (!has_costfun) std::cout << "(NO COSTFUN) ";
      if (!has_dynamics) std::cout << "(NO DYNAMICS) ";
      if (!has_initial_state) std::cout << "(NO INITIAL STATE) ";
      if (!good_state_dim) std::cout << "(INCONSISTENT INITIAL STATE)";
      std::cout << std::endl;
    }
  }
  return valid;
}

// Specialize the method for each constraint type
template <>
void Problem::SetConstraint<constraints::Equality>(
    std::shared_ptr<constraints::Constraint<constraints::Equality>> con, int k) {
  ALTRO_ASSERT(con != nullptr, "Must provide a valid constraint pointer.");
  ALTRO_ASSERT(con->OutputDimension() > 0, "Constraint must have a length greater than zero.");
  eq_[k].emplace_back(std::move(con));
}

template <>
void Problem::SetConstraint<constraints::Inequality>(
    std::shared_ptr<constraints::Constraint<constraints::Inequality>> con, int k) {
  ALTRO_ASSERT(con != nullptr, "Must provide a valid constraint pointer.");
  ALTRO_ASSERT(con->OutputDimension() > 0, "Constraint must have a length greater than zero.");
  ineq_[k].emplace_back(std::move(con));
}

}  // namespace problem
}  // namespace altro