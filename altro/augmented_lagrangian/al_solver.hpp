#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <algorithm>
#include <limits>
#include <type_traits>

#include "altro/augmented_lagrangian/al_problem.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/constraints/constraint_values.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace augmented_lagrangian {

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
  explicit AugmentedLagrangianiLQR(int N) : ilqr_solver_(N), costs_(), max_violation_(N + 1) {}
  explicit AugmentedLagrangianiLQR(const problem::Problem& prob);

  void InitializeFromProblem(const problem::Problem& prob);

  /***************************** Getters **************************************/

  SolverStats& GetStats() { return ilqr_solver_.GetStats(); }
  const SolverStats& GetStats() const { return ilqr_solver_.GetStats(); }
  SolverOptions& GetOptions() { return ilqr_solver_.GetOptions(); }
  const SolverOptions& GetOptions() const { return ilqr_solver_.GetOptions(); }
  SolverStatus GetStatus() const { return status_; }
  std::shared_ptr<ALCost<n, m>> GetALCost(const int k) { return costs_.at(k); }
  ilqr::iLQR<n, m>& GetiLQRSolver() { return ilqr_solver_; }
  int NumSegments() const { return ilqr_solver_.NumSegments(); }

  int NumConstraints(const int& k) const;
  int NumConstraints() const;

  /**
   * @brief Print a summary of all of the constraints in the problem.
   * 
   * Prints the label of each constraint, it's knot point index, and a vector 
   * of it's violations.
   * 
   * Each element contains a basic description of the constraint, the knot
   * point index at which it's located, and a vector of it's current violations.
   * 
   * @param[in] should_sort Sorts the constraints by the infinity norm of their 
   *                        constraint. Default is false (no sorting), where 
   *                        they are sorted by knot point index.
   * @param[in] precision   Controls the precision of the numerical output.
   */
  void PrintViolations(bool should_sort = false, int precision = 4) {
    std::vector<constraints::ConstraintInfo> coninfo = GetConstraintInfo(should_sort);
    fmt::print("Got {} constraints\n", coninfo.size());
    for (const constraints::ConstraintInfo& info : coninfo) {
      fmt::print("{}\n", info.ToString(precision));
    }
  }

  /**
   * @brief Get a list of the constraints.
   * 
   * Each element contains a basic description of the constraint, the knot
   * point index at which it's located, and a vector of it's current violations.
   * 
   * @param should_sort Sorts the constraints by the infinity norm of their constraint 
   * violations.
   * @return std::vector<constraints::ConstraintInfo>
   */
  std::vector<constraints::ConstraintInfo> GetConstraintInfo(bool should_sort = false) const {
    std::vector<constraints::ConstraintInfo> coninfo;
    int i_last = 0;
    for (int k = 0; k <= NumSegments(); ++k) {
      costs_[k]->GetConstraintInfo(&coninfo);
      for (int i = i_last; i < static_cast<int>(coninfo.size()); ++i) {
        coninfo[i].index = k;
      }
      i_last = coninfo.size();
    }
    if (should_sort) {
      auto comp = [](const constraints::ConstraintInfo& info1,
                     const constraints::ConstraintInfo& info2) {
        return info1.violation.lpNorm<Eigen::Infinity>() > info2.violation.lpNorm<Eigen::Infinity>();
      };
      std::sort(coninfo.begin(), coninfo.end(), comp);
    }
    return coninfo;
  }

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
  void SetPenalty(const double& rho);

  /**
   * @brief Set the Penalty scaling parameter to be the same for all constraints
   * and knot points.
   *
   * To set the penalty scaling independently for different constraints and/or
   * knot points, use GetALCost(k).SetPenaltyScaling<ConType>(phi, i).
   *
   * @param phi Penalty parameter (phi > 1).
   */
  void SetPenaltyScaling(const double& phi);

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

  void Init();

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

  void ResetDualVariables();

 private:
  Stopwatch CreateTimer(const std::string& name) { return GetStats().GetTimer()->Start(name); }

  ilqr::iLQR<n, m> ilqr_solver_;
  std::vector<std::shared_ptr<ALCost<n, m>>> costs_;
  SolverStatus status_ = SolverStatus::kUnsolved;
  VectorXd max_violation_;  // (N+1,) vector of constraint violations at each knot point
};

////////////////////////////////////////////////////////////////////////////////
/**************************** Implementation **********************************/
////////////////////////////////////////////////////////////////////////////////

template <int n, int m>
AugmentedLagrangianiLQR<n, m>::AugmentedLagrangianiLQR(const problem::Problem& prob)
    : ilqr_solver_(prob.NumSegments()),
      costs_(),
      max_violation_(VectorXd::Zero(prob.NumSegments() + 1)) {
  InitializeFromProblem(prob);
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::InitializeFromProblem(const problem::Problem& prob) {
  max_violation_.setZero(prob.NumSegments() + 1);
  problem::Problem prob_al = BuildAugLagProblem<n, m>(prob, &costs_);
  ALTRO_ASSERT(static_cast<int>(costs_.size()) == prob.NumSegments() + 1,
               fmt::format("Got an incorrect number of cost functions. Expected {}, got {}",
                           prob.NumSegments(), costs_.size()));
  // ilqr_solver_.InitializeFromProblem(prob_al);
  ilqr_solver_.CopyFromProblem(prob_al, 0, prob.NumSegments() + 1);
  auto max_violation_callback = [this]() -> double { return this->GetMaxViolation(); };
  ilqr_solver_.SetConstraintCallback(max_violation_callback);
}

template <int n, int m>
int AugmentedLagrangianiLQR<n, m>::NumConstraints(const int& k) const {
  ALTRO_ASSERT(0 <= k && k <= NumSegments(),
               fmt::format("Invalid knot point index. Got {}, expected to be in range [{},{}]", k,
                           0, NumSegments()));
  ALTRO_ASSERT(
      static_cast<int>(costs_.size()) == NumSegments() + 1,
      "Cannot query the number of constraints before initializing the solver with a problem.");
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
void AugmentedLagrangianiLQR<n, m>::SetPenalty(const double& rho) {
  for (int k = 0; k <= NumSegments(); ++k) {
    costs_[k]->template SetPenalty<constraints::Equality>(rho);
    costs_[k]->template SetPenalty<constraints::Inequality>(rho);
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::SetPenaltyScaling(const double& phi) {
  for (int k = 0; k <= NumSegments(); ++k) {
    costs_[k]->template SetPenaltyScaling<constraints::Equality>(phi);
    costs_[k]->template SetPenaltyScaling<constraints::Inequality>(phi);
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::Init() {
  Stopwatch sw = CreateTimer("init");

  SolverStats& stats = GetStats();
  if (GetOptions().reset_duals) {
    ResetDualVariables();
  }
  if (GetOptions().initial_penalty > 0) {
    SetPenalty(GetOptions().initial_penalty);
  }
  stats.Reset();
  stats.Log("iter_al", 0);
  stats.Log("viol", MaxViolation());
  stats.Log("pen", GetMaxPenalty());
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::Solve() {
  // This check needs to happen before creating the first stopwatch
  GetOptions().profiler_enable ? GetStats().GetTimer()->Activate()
                               : GetStats().GetTimer()->Deactivate();
  Stopwatch sw = CreateTimer("al");

  Init();

  for (int iteration = 0; iteration < GetOptions().max_iterations_outer; ++iteration) {
    ilqr_solver_.Solve();
    UpdateDuals();
    UpdateConvergenceStatistics();

    // Print the log data here if iLQR isn't printing it
    bool is_ilqr_logging = GetStats().GetVerbosity() >= LogLevel::kInner;
    if (!is_ilqr_logging) {
      GetStats().PrintLast();
    }

    if (IsDone()) {
      break;
    }

    // If iLQR is printing the logs, print the header before every new AL iteration
    if (is_ilqr_logging) {
      GetStats().GetLogger().PrintHeader();
    }
    UpdatePenalties();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdateDuals() {
  Stopwatch sw = CreateTimer("dual_update");

  int N = this->NumSegments();
  for (int k = 0; k <= N; ++k) {
    // fmt::print("Updating Duals at index {}...\n", k);
    costs_[k]->UpdateDuals();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdatePenalties() {
  Stopwatch sw = CreateTimer("penalty_update");

  int N = this->NumSegments();
  for (int k = 0; k <= N; ++k) {
    costs_[k]->UpdatePenalties();
  }
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::UpdateConvergenceStatistics() {
  Stopwatch sw = CreateTimer("stats");

  SolverStats& stats = GetStats();
  stats.iterations_outer++;
  stats.Log("viol", GetMaxViolation());
  stats.Log("pen", GetMaxPenalty());
  stats.Log("iter_al", stats.iterations_outer);
}

template <int n, int m>
bool AugmentedLagrangianiLQR<n, m>::IsDone() {
  Stopwatch sw = CreateTimer("convergence_check");

  SolverStats& stats = GetStats();
  SolverOptions& opts = GetOptions();
  const bool are_constraints_satisfied = stats.violations.back() < opts.constraint_tolerance;
  const bool is_max_penalty_exceeded = stats.max_penalty.back() > opts.maximum_penalty;
  const bool is_max_outer_iterations_exceeded = stats.iterations_outer >= opts.max_iterations_outer;
  const bool is_max_total_iterations_exeeded = stats.iterations_total >= opts.max_iterations_total;
  if (ilqr_solver_.GetStatus() != SolverStatus::kSolved) {
    status_ = ilqr_solver_.GetStatus();
    return true;
  }
  if (are_constraints_satisfied) {
    if (ilqr_solver_.GetStatus() == SolverStatus::kSolved) {
      status_ = SolverStatus::kSolved;
      return true;
    }
  }
  if (is_max_penalty_exceeded) {
    status_ = SolverStatus::kMaxPenalty;
    return true;
  }
  if (is_max_outer_iterations_exceeded) {
    status_ = SolverStatus::kMaxOuterIterations;
    return true;
  }
  if (is_max_total_iterations_exeeded) {
    status_ = SolverStatus::kMaxIterations;
    return true;
  }
  return false;
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
    max_violation_(k) = costs_[k]->template MaxViolation<p>();
  }
  return max_violation_.template lpNorm<p>();
}

template <int n, int m>
double AugmentedLagrangianiLQR<n, m>::GetMaxPenalty() const {
  double max_penalty = 0.0;

  for (int k = 0; k <= NumSegments(); ++k) {
    max_penalty = std::max(max_penalty, costs_[k]->MaxPenalty());
  }
  return max_penalty;
}

template <int n, int m>
void AugmentedLagrangianiLQR<n, m>::ResetDualVariables() {
  for (auto& alcost : costs_) {
    alcost->ResetDualVariables();
  }
}

}  // namespace augmented_lagrangian
}  // namespace altro
