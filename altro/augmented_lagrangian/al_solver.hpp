#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <type_traits>

#include "altro/augmented_lagrangian/al_problem.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace augmented_lagrangian {

struct AugmentedLagrangianOptions {
  int max_iterations_outer = 30;   // Maximum augmented Lagrangian iterations
  int max_iterations_total = 300;  // Maximum number of total iterative LQR iterations

  double constraint_tolerance = 1e-4;  // Maximum constraint violation theshold
  double maximum_penalty = 1e8;        // Maximum penalty parameter allowed
  int verbose = 0;                     // Output verbosity level
  // TODO(bjackson): Implement better vebose output and document
};

struct AugmentedLagrangianStats {
  int iterations_outer = 0;           // Number of augmented Lagrangian updates / iLQR solves
  int iterations_total = 0;           // Total number of iLQR iterations
  std::vector<double> violations;     // The maximum constraint violation for each AL iteration
  std::vector<double> max_penalty;    // Maximum penalty parameter for each AL iteration
  std::vector<int> inner_iterations;  // iLQR iterations for each AL iteration
};

/**
 * @brief Trajectory optimization solver that uses augmented Lagrangian to
 * handle arbitrary constraints, while using DDP / iLQR to solve the resulting
 * unconstrained trajectory optimization problem.
 *
 * @tparam n Compile-time state dimension.
 * @tparam m Compile-time control dimension.
 */
template <int n, int m>
class AugmentedLagrangianiLQR {
  template <class ConType>
  using ConstraintValueVec =
      std::vector<std::shared_ptr<constraints::ConstraintValues<n, m, ConType>>>;

 public:
  explicit AugmentedLagrangianiLQR(const problem::Problem& prob);

  /***************************** Getters **************************************/

  AugmentedLagrangianStats& GetStats() { return stats_; }
  const AugmentedLagrangianStats& GetStats() const { return stats_; }
  AugmentedLagrangianOptions& GetOptions() { return opts_; }
  const AugmentedLagrangianOptions& GetOptions() const { return opts_; }
  ilqr::SolverStatus GetStatus() const { return status_; }
  std::shared_ptr<ALCost<n, m>> GetALCost(const int k) { return costs_.at(k); }
  ilqr::iLQR<n, m>& GetiLQRSolver() { return ilqr_solver_; }
  int NumSegments() const { return ilqr_solver_.NumSegments(); }

  int NumConstraints(const int k) const;
  int NumConstraints() const;

  /***************************** Setters **************************************/

  /**
   * @brief Set the Penalty parameter to be the same for all constraints and
   * knot points.
   *
   * To set the penalty independently for different constraints and/or
   * knot points, use GetALCost(k).SetPenalty<ConType>(rho, i).
   *
   * @param rho
   */
  void SetPenalty(const double rho);

  /**
   * @brief Set the Penalty scaling parameter to be the same for all constraints
   * and knot points.
   *
   * To set the penalty scaling independently for different constraints and/or
   * knot points, use GetALCost(k).SetPenaltyScaling<ConType>(phi, i).
   *
   * @param phi Penalty parameter (phi > 1).
   */
  void SetPenaltyScaling(const double phi);

  /**
   * @brief Specify the initial guess for the state and control trajectory.
   *
   * This trajectory will be modifed by the solve and will be equal to the
   * optimized trajectory after the solve is complete.
   *
   * @param traj A pointer to the trajectory.
   */
  void SetTrajectory(std::shared_ptr<Trajectory<n, m>> traj) {
    ilqr_solver_.SetTrajectory(std::move(traj));
  }

  /***************************** Methods **************************************/

  /**
   * @brief Solve the trajectory optimization problem using AL-iLQR.
   *
   */
  void Solve();

  /**
   * @brief Update the dual variables for all of the constraints
   *
   */
  void UpdateDuals();

  /**
   * @brief Update the penalty parameters for all of the constraints
   *
   */
  void UpdatePenalties();

  /**
   * @brief Calculate the convergence criterion for augmented Lagrangian
   *
   */
  void UpdateConvergenceStatistics();

  /**
   * @brief Checks if the solve can terminate.
   *
   * Will terminate either because it has met the convergence criteria, or it
   * has failed in some way.
   *
   * @return true if the solver should stop iterating.
   */
  bool IsDone();

  /**
   * @brief Print a summary of the last iteration.
   *
   */
  void PrintLast() const;

  /**
   * @brief Calculate the maximum constraint violation.
   * Updates the constraints by evaluating the augmented Lagrangian cost.
   *
   * @tparam p Norm to use to calculate violation (default is infinity).
   * @param[in] Z The trajectory to use to evaluate the constraints. Defaults
   * to the trajectory stored by the internal iLQR solver.
   * @return Maximum constraint violation. Should be close to zero if
   * the solve is successful.
   */
  template <int p = Eigen::Infinity>
  double MaxViolation();

  template <int p = Eigen::Infinity>
  double MaxViolation(const Trajectory<n, m>& Z);

  /**
   * @brief Calculate the maximum constraint violation without calculating
   * the cost. It will use the currently stored constraint values.
   *
   * @tparam p Norm to use to calculate violation (default is infinity).
   * @return Maximum constraint violation. Should be close to zero if
   * the solve is successful.
   */
  template <int p = Eigen::Infinity>
  double GetMaxViolation();

  /**
   * @brief Get the maximum penalty parameter used across all constraints and
   * knot points.
   *
   * @return The maximum penalty parameter.
   */
  double GetMaxPenalty() const;

 private:
  ilqr::iLQR<n, m> ilqr_solver_;
  AugmentedLagrangianOptions opts_;
  AugmentedLagrangianStats stats_;
  std::vector<std::shared_ptr<ALCost<n, m>>> costs_;
  ilqr::SolverStatus status_ = ilqr::SolverStatus::kUnsolved;
  VectorXd max_violation_;  // (N+1,) vector of constraint violations at each knot point
};

////////////////////////////////////////////////////////////////////////////////
/**************************** Implementation **********************************/
////////////////////////////////////////////////////////////////////////////////

template <int n, int m>
AugmentedLagrangianiLQR<n, m>::AugmentedLagrangianiLQR(const problem::Problem& prob)
    : ilqr_solver_(prob.NumSegments()),
      opts_(),
      stats_(),
      costs_(),
      max_violation_(VectorXd::Zero(prob.NumSegments() + 1)) {
  problem::Problem prob_al = BuildAugLagProblem<n, m>(prob, &costs_);
  ALTRO_ASSERT(static_cast<int>(costs_.size()) == prob.NumSegments() + 1,
               fmt::format("Got an incorrect number of cost functions. Expected {}, got {}",
                           prob.NumSegments(), costs_.size()));
  ilqr_solver_.SetInitialState(prob_al.GetInitialState());
  ilqr_solver_.CopyFromProblem(prob_al, 0, prob.NumSegments() + 1);
}

template <int n, int m>
int AugmentedLagrangianiLQR<n, m>::NumConstraints(const int k) const {
  ALTRO_ASSERT(0 <= k && k <= NumSegments(),
               fmt::format("Invalid knot point index. Got {}, expected to be in range [{},{}]", k,
                           0, NumSegments()));
  return costs_.at(k)->NumConstraints();
}

template <int n, int m>
int AugmentedLagrangianiLQR<n, m>::NumConstraints() const {
  int cnt = 0;
  for (int k = 0; k <= NumSegments(); ++k) {
    cnt += NumConstraints(k);
  }
  return cnt;
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::SetPenalty(const double rho) {
  for (int k = 0; k <= NumSegments(); ++k) {
    costs_[k]->template SetPenalty<constraints::Equality>(rho);
    costs_[k]->template SetPenalty<constraints::Inequality>(rho);
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::SetPenaltyScaling(const double phi) {
  for (int k = 0; k <= NumSegments(); ++k) {
    costs_[k]->template SetPenaltyScaling<constraints::Equality>(phi);
    costs_[k]->template SetPenaltyScaling<constraints::Inequality>(phi);
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::Solve() {
  for (int iteration = 0; iteration < opts_.max_iterations_outer; ++iteration) {
    ilqr_solver_.Solve();
    UpdateDuals();
    UpdateConvergenceStatistics();
    if (opts_.verbose) {
      PrintLast();
    }
    if (IsDone()) {
      break;
    }
    UpdatePenalties();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdateDuals() {
  int N = this->NumSegments();
  for (int k = 0; k <= N; ++k) {
    // fmt::print("Updating Duals at index {}...\n", k);
    costs_[k]->UpdateDuals();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdatePenalties() {
  int N = this->NumSegments();
  for (int k = 0; k <= N; ++k) {
    costs_[k]->UpdatePenalties();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdateConvergenceStatistics() {
  const int inner_iters = ilqr_solver_.GetStats().iterations;
  stats_.iterations_total += inner_iters;
  stats_.inner_iterations.emplace_back(inner_iters);
  stats_.iterations_outer++;
  stats_.violations.emplace_back(GetMaxViolation());
  stats_.max_penalty.emplace_back(GetMaxPenalty());
}

template <int n, int m>
bool AugmentedLagrangianiLQR<n, m>::IsDone() {
  const bool are_constraints_satisfied = stats_.violations.back() < opts_.constraint_tolerance;
  const bool is_max_penalty_exceeded = stats_.max_penalty.back() > opts_.maximum_penalty;
  const bool is_max_outer_iterations_exceeded = stats_.iterations_outer >= opts_.max_iterations_outer;
  const bool is_max_total_iterations_exeeded = stats_.iterations_total >= opts_.max_iterations_total;
  if (are_constraints_satisfied) {
    if (ilqr_solver_.GetStatus() == ilqr::SolverStatus::kSolved) {
      status_ = ilqr::SolverStatus::kSolved;
      return true;
    }
  }
  if (is_max_penalty_exceeded) {
    status_ = ilqr::SolverStatus::kMaxPenalty;
    return true;
  }
  if (is_max_outer_iterations_exceeded) {
    status_ = ilqr::SolverStatus::kMaxOuterIterations;
    return true;
  }
  if (is_max_total_iterations_exeeded) {
    status_ = ilqr::SolverStatus::kMaxIterations;
    return true;
  }
  return false;
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::PrintLast() const {
  fmt::print("Iter {:3>}: Cost = {:.3}, Viol = {:.4e}\n", stats_.iterations_total,
             ilqr_solver_.GetStats().cost.back(), stats_.violations.back());
}

template <int n, int m>
template <int p>
double AugmentedLagrangianiLQR<n, m>::MaxViolation() {
  ilqr_solver_.Cost();  // Calculate cost to update constraints
  return GetMaxViolation<p>();
}

template <int n, int m>
template <int p>
double AugmentedLagrangianiLQR<n, m>::MaxViolation(const Trajectory<n, m>& Z) {
  ilqr_solver_.Cost(Z);  // Calculate cost to update constraints
  return GetMaxViolation<p>();
}

template <int n, int m>
template <int p>
double AugmentedLagrangianiLQR<n, m>::GetMaxViolation() {
  for (int k = 0; k <= NumSegments(); ++k) {
    max_violation_(k) = costs_[k]->MaxViolation<p>();
  }
  return max_violation_.lpNorm<p>();
}

template <int n, int m>
double AugmentedLagrangianiLQR<n, m>::GetMaxPenalty() const {
  double max_penalty = 0.0;
  for (int k = 0; k <= NumSegments(); ++k) {
    max_penalty = std::max(max_penalty, costs_[k]->MaxPenalty());
  }
  return max_penalty;
}

}  // namespace augmented_lagrangian
}  // namespace altro
