#pragma once

#include <iostream>
#include <map>

#include "altro/common/state_control_sized.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"

namespace altro {
namespace ilqr {

/**
 * @brief Solver options for iLQR
 *
 */
struct iLQROptions {
  // clang-format off
  int max_iterations = 100;               // Max iterations before exiting
  double cost_tolerance = 1e-4;           // Threshold for cost decrease 
  double gradient_tolerance = 1e-2;       // Threshold for infinity-norm of the approximate gradient

  double bp_reg_increase_factor = 1.6;    // Multiplicative factor for increasing the regularization
  double bp_reg_enable = true;            // Enable regularization in the backward pass
  double bp_reg_initial = 0.0;            // Initial regularization
  double bp_reg_max = 1e8;                // Maximum regularization
  double bp_reg_min = 1e-8;               // Minimum regularization
  // double bp_reg_forwardpass = 10.0;     
  int bp_reg_fail_threshold = 100;        // How many time the backward pass can fail before throwing an error
  bool check_forwardpass_bounds = true;   // Whether to check if the rollouts stay within the specified bounds
  double state_max = 1e8;                 // Maximum state value (abs)
  double control_max = 1e8;               // Maximum control value (abs)

  int line_search_max_iterations = 20;    // Maximum number of line search iterations before increasing regularization
  double line_search_lower_bound = 1e-8;  // Sufficient improvement condition
  double line_search_upper_bound = 10.0;  // Can't make too much more improvement than expected
  double line_search_decrease_factor = 2; // How much the line search step size is decreased each iteration
  int verbose = 0;                        // Control output level
  // clang-format on
};

/**
 * @brief Records statistics during a single iLQR solve
 *
 * Captures important information like iterations, and initial cost,
 * and critical per-iteration information like cost, gradient, cost decrease
 * etc.
 */
struct iLQRStats {
  double initial_cost = 0.0;
  int iterations = 0;
  std::vector<double> cost;
  std::vector<double> alpha;
  std::vector<double> improvement_ratio;  // ratio of actual to expected cost decrease
  std::vector<double> gradient;
  std::vector<double> cost_decrease;
  std::vector<double> regularization;

  void SetCapacity(int n) {
    cost.reserve(n);
    alpha.reserve(n);
    improvement_ratio.reserve(n);
    gradient.reserve(n);
    cost_decrease.reserve(n);
  }

  void Reset() {
    int cur_capacity = cost.capacity();
    cost.clear();
    alpha.clear();
    improvement_ratio.clear();
    gradient.clear();
    cost_decrease.clear();
    SetCapacity(cur_capacity);
  }

  void PrintLast() {
    // clang-format off
    std::cout << "Iter " << iterations << ": \tCost = " << cost.back()
              << "\t dJ = " << cost_decrease.back() 
              << "\t z = " << improvement_ratio.back() 
              << "\t alpha = " << alpha.back()
              << "\tgrad = " << gradient.back() << std::endl;
    // clang-format on
  }
};

/**
 * @brief Describes the current state of the solver
 *
 * Used to describe if the solver successfully solved the problem or to
 * provide a reason why it was unsuccessful.
 */
enum class SolverStatus {
  kSolved = 0,
  kUnsolved = 1,
  kStateLimit,
  kControlLimit,
  kCostIncrease,
  kMaxIterations,
};

/**
 * @brief Solve an unconstrained trajectory optimization problem using
 * iterative LQR.
 *
 * The iLQR algorithm works taking a second-order approximation of the cost
 * function and a first-order expansion of the dynamics. A locally-optimal
 * feedback control policy is then constructed around the current estimate
 * of the optimal trajectory, which is calculated using a generalization of
 * time-varying LQR during the "backward pass". This policy is then used to
 * simulate the system forward during the "forward pass", and the process
 * is repeated until convergence. Since the system is simulated forward
 * every iteration, iLQR effectively only optimizes directly over the
 * control variables.
 *
 * @tparam n Compile-time state dimension.
 * @tparam m Compile-time control dimension.
 */
template <int n = Eigen::Dynamic, int m = Eigen::Dynamic>
class iLQR {
 public:
  explicit iLQR(int N) : N_(N), opts_(), knotpoints_() { Init(); }
  explicit iLQR(const problem::Problem& prob)
      : N_(prob.NumSegments()), initial_state_(prob.GetInitialState()) {
    CopyFromProblem(prob, 0, N_ + 1);
    Init();
  }

  /**
   * @brief Copy the data from a Problem class into the iLQR solver
   *
   * Capture shared pointers to the cost and dynamics objects for each
   * knot point, storing them in the correspoding KnotPointFunctions object.
   *
   * Assumes both the problem and the solver have the number of knot points.
   *
   * Allows for a subset of the knot points to be copied, since in the future
   * this method might be used to specify compile-time sizes for hybrid /
   * switched dynamics.
   *
   * @tparam n2 Compile-time state dimension. Can be Eigen::Dynamic (-1)
   * @tparam m2 Compile-time control dimension. Can be Eigen::Dynamic (-1)
   * @param prob Trajectory optimization problem
   * @param k_start Starting index (inclusive) for data to copy
   * @param k_stop Terminal index (exclusive) for data to copy
   */
  template <int n2 = n, int m2 = m>
  void CopyFromProblem(const problem::Problem& prob, int k_start, int k_stop) {
    ALTRO_ASSERT(prob.IsFullyDefined(), "Expected problem to be fully defined.");
    int state_dim = 0;
    int control_dim = 0;
    for (int k = k_start; k < k_stop; ++k) {
      std::shared_ptr<problem::DiscreteDynamics> model = prob.GetDynamics(k);
      std::shared_ptr<problem::CostFunction> costfun = prob.GetCostFunction(k);

      // Model will be nullptr at the last knot point
      if (model) {
        state_dim = model->StateDimension();
        control_dim = model->ControlDimension();
        knotpoints_.emplace_back(std::make_unique<ilqr::KnotPointFunctions<n2, m2>>(model, costfun));
      } else {
        // To construct the KPF at the terminal knot point we need to tell
        // it the state and control dimensions since we don't have a dynamics
        // function
        ALTRO_ASSERT(k == N_, "Expected model to only be a nullptr at last time step");
        ALTRO_ASSERT(state_dim != 0 && control_dim != 0,
                     "The last time step cannot be copied in isolation. "
                     "Include the previous time step, e.g. "
                     "CopyFromProblem(prob,N-1,N+1)");
        knotpoints_.emplace_back(
            std::make_unique<ilqr::KnotPointFunctions<n, m>>(state_dim, control_dim, costfun));
      }
    }
  }

  /***************************** Getters **************************************/
  /**
   * @brief Get a pointer to the trajectory
   *
   */
  std::shared_ptr<Trajectory<n, m>> GetTrajectory() { return Z_; }

  /**
   * @brief Return the number of segments in the trajectory
   */
  int NumSegments() const { return N_; }
  /**
   * @brief Get the Knot Point Function object, which contains all of the
   * data for each knot point, including cost and dynamics expansions,
   * feedback and feedforward gains, cost-to-go expansion, etc.
   *
   * @param k knot point index, 0 <= k <= N_
   * @return reference to the KnotPointFunctions class
   */
  KnotPointFunctions<n, m>& GetKnotPointFunction(int k) {
    ALTRO_ASSERT((k >= 0) && (k <= N_), "Invalid knot point index.");
    return *(knotpoints_[k]);
  }

  iLQRStats& GetStats() { return stats_; }
  iLQROptions& GetOptions() { return opts_; }
  VectorXd& GetCosts() { return costs_; }
  SolverStatus GetStatus() const { return status_; }
  VectorNd<n>& GetInitialState() { return initial_state_; }
  double GetRegularization() { return rho_; }

  /***************************** Setters **************************************/
  /**
   * @brief Store a pointer to the trajectory
   *
   * This trajectory will be used as the initial guess and will also be the
   * storage location for the optimized trajectory.
   *
   * @param traj Pointer to the trajectory
   */
  void SetTrajectory(std::shared_ptr<Trajectory<n, m>> traj) {
    Z_ = std::move(traj);
    Zbar_ = std::make_unique<Trajectory<n, m>>(*Z_);
    Zbar_->SetZero();
  }

  /***************************** Algorithm **************************************/
  /**
   * @brief Solve the trajectory optimization problem using iLQR
   *
   * @post The provided trajectory is overwritten with a locally-optimal
   * dynamically-feasible trajectory. The solver status and statistics,
   * obtained via GetStatus() and GetStats() are updated.
   * The solve is successful if `GetStatus == ilqr::SolverStatus::kSuccess`.
   *
   */
  void Solve() {
    Initialize();  // reset any internal variables
    Rollout();     // simulate the system forward using initial controls
    stats_.initial_cost = Cost();

    for (int iter = 0; iter < opts_.max_iterations; ++iter) {
      UpdateExpansions();
      BackwardPass();
      ForwardPass();
      UpdateConvergenceStatistics();
      if (opts_.verbose >= 1) {
        stats_.PrintLast();
      }
      if (IsDone()) {
        break;
      }
    }

    WrapUp();
  }

  /**
   * @brief Calculate the cost of the current trajectory
   *
   * By default, it will use the current guess stored in the solver, but it
   * can be passed any compatible trajectory.
   *
   * @return double The current cost
   */
  double Cost() { return Cost(*Z_); }
  double Cost(const Trajectory<n, m>& Z) {
    CalcIndividualCosts(Z);
    return costs_.sum();
  }

  /**
   * @brief Update the cost and dynamics expansions
   *
   * NOTE: Also calculates the cost for each knot point.
   *
   * Computes the first and second order expansions of the cost and dynamics,
   * storing the results in the KnotPointFunctions class for each knot point.
   *
   * @pre The trajectory must set to the next guess for the optimal trajectory.
   * The trajectory cannot be a nullptr, and must be set via SetTrajectory.
   *
   * @post The expansions are updated for knotpoints_[k], 0 <= k < N_
   *
   */
  void UpdateExpansions() {
    ALTRO_ASSERT(Z_ != nullptr, "Trajectory pointer must be set before updating the expansions.");

    // TODO(bjackson): do this in parallel
    for (int k = 0; k <= N_; ++k) {
      KnotPoint<n, m>& z = Z_->GetKnotPoint(k);
      knotpoints_[k]->CalcCostExpansion(z.State(), z.Control());
      knotpoints_[k]->CalcDynamicsExpansion(z.State(), z.Control(), z.GetTime(), z.GetStep());
      costs_(k) = GetKnotPointFunction(k).Cost(z.State(), z.Control());
    }
  }

  /**
   * @brief Compute a locally optimal linear-feedback policy
   *
   * The backward pass uses time-varying LQR to compute an optimal
   * linear-feedback control policy. As the solve converges the constant
   * feed-forward terms should go to zero. The solve also computes a local
   * quadratic approximation of the cost-to-go.
   *
   * @pre The cost and dynamics expansions have already been computed using
   * UpdateExpansions.
   *
   * @post The feedforward and feedback gains, action-value expansion, and
   * cost-to-go expansion terms are all updated inside the KnotPointFunctions
   * class for each knot point. The overall expected cost decrease is stored
   * in deltaV_.
   *
   */
  void BackwardPass() {
    // Regularization
    Eigen::ComputationInfo info;

    // Terminal Cost-to-go
    knotpoints_[N_]->CalcTerminalCostToGo();
    Eigen::Matrix<double, n, n>* Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
    Eigen::Matrix<double, n, 1>* Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

    int max_reg_count = 0;
    deltaV_[0] = 0.0;
    deltaV_[1] = 0.0;
    for (int k = N_ - 1; k >= 0; --k) {
      knotpoints_[k]->CalcActionValueExpansion(*Sxx_prev, *Sx_prev);
      knotpoints_[k]->RegularizeActionValue(rho_);
      info = knotpoints_[k]->CalcGains();

      // Handle solve failure
      if (info != Eigen::Success) {
        std::cout << "Failed solve at knot point" << k << std::endl;
        IncreaseRegularization();
        k = N_ - 1;  // Start at the beginning of the trajectory again

        // Check if we're at max regularization
        if (rho_ >= opts_.bp_reg_max) {
          max_reg_count++;
        }

        // Throw an error if we keep failing, even at max regularization
        // TODO(bjackson): Look at better ways of doing this
        if (max_reg_count >= opts_.bp_reg_fail_threshold) {
          throw std::runtime_error("Backward pass regularization increased too many times.");
        }
        continue;
      }

      // Update Cost-To-Go
      knotpoints_[k]->CalcCostToGo();
      knotpoints_[k]->AddCostToGo(deltaV_);

      Sxx_prev = &(knotpoints_[k]->GetCostToGoHessian());
      Sx_prev = &(knotpoints_[k]->GetCostToGoGradient());
    }
    stats_.regularization.push_back(rho_);
    DecreaseRegularization();
  }

  /**
   * @brief Simulate the dynamics forward from the initial state
   *
   * By default it will simulate the system forward open-loop.
   *
   */
  void Rollout() {
    Z_->State(0) = initial_state_;
    for (int k = 0; k < N_; ++k) {
      knotpoints_[k]->Dynamics(Z_->State(k), Z_->Control(k), Z_->GetTime(k), Z_->GetStep(k),
                               Z_->State(k + 1));
    }
  }

  /**
   * @brief Simulate the system forward using the feedback and feedforward
   * gains calculated during the backward pass.
   *
   * @param alpha Line search parameter, 0 < alpha <= 1.
   * @return true If the the state and control bounds are not violated.
   */
  bool RolloutClosedLoop(const double alpha) {
    Zbar_->State(0) = initial_state_;
    for (int k = 0; k < N_; ++k) {
      MatrixNxMd<m, n>& K = GetKnotPointFunction(k).GetFeedbackGain();
      VectorNd<m>& d = GetKnotPointFunction(k).GetFeedforwardGain();

      // TODO(bjackson): Make this a function of the dynamics
      VectorNd<n> dx = Zbar_->State(k) - Z_->State(k);
      Zbar_->Control(k) = Z_->Control(k) + K * dx + d * alpha;

      // Simulate forward with feedback
      GetKnotPointFunction(k).Dynamics(Zbar_->State(k), Zbar_->Control(k), Zbar_->GetTime(k),
                                       Zbar_->GetStep(k), Zbar_->State(k + 1));

      if (opts_.check_forwardpass_bounds) {
        if (Zbar_->State(k + 1).norm() > opts_.state_max) {
          // TODO(bjackson): Emit warning (need logging mechanism)
          status_ = SolverStatus::kStateLimit;
          return false;
        }
        if (Zbar_->Control(k).norm() > opts_.control_max) {
          // TODO(bjackson): Emit warning (need logging mechanism)
          status_ = SolverStatus::kControlLimit;
          return false;
        }
      }
    }
    status_ = SolverStatus::kUnsolved;
    return true;
  }

  /**
   * @brief Attempt to find a better state-control trajectory
   *
   * Using the feedback policy computed during the backward pass,
   * simulate the system forward and make sure the resulting trajectory
   * decreases the overall cost and make sufficient progress towards a
   * local minimum (via pseudo Wolfe conditions).
   *
   * @post The current trajectory candidate Z_ is updated with the new guess.
   *
   */
  void ForwardPass() {
    double J0 = costs_.sum();  // Calculated during UpdateExpansions

    double alpha = 1.0;
    double z = -1.0;
    int iter_fp = 0;
    bool success = false;

    double J = J0;

    for (; iter_fp < opts_.line_search_max_iterations; ++iter_fp) {
      if (RolloutClosedLoop(alpha)) {
        J = Cost(*Zbar_);
        double expected = -alpha * (deltaV_[0] + alpha * deltaV_[1]);
        if (expected > 0.0) {
          z = (J0 - J) / expected;
        } else {
          z = -1.0;
        }

        if (opts_.line_search_lower_bound <= z && z <= opts_.line_search_upper_bound && J < J0) {
          success = true;
          stats_.alpha.emplace_back(alpha);
          stats_.improvement_ratio.emplace_back(z);
          stats_.cost.emplace_back(J);
          break;
        }
      }
      alpha /= opts_.line_search_decrease_factor;
    }

    if (success) {
      (*Z_) = (*Zbar_);
    } else {
      IncreaseRegularization();
      J = J0;
    }

    if (J > J0) {
      // TODO(bjackson): Emit warning (needs logging)
      status_ = SolverStatus::kCostIncrease;
    }
  }

  /**
   * @brief Evaluate all the information necessary to check convergence
   *
   * Calculates the gradient, change in cost, etc. Updates the solver statistics
   * accordingly.
   *
   * @post Increments the number of solver iterations
   */
  void UpdateConvergenceStatistics() {
    double dgrad = NormalizedFeedforwardGain();
    double dJ = stats_.cost.rbegin()[1] - stats_.cost.rbegin()[0];

    stats_.gradient.emplace_back(dgrad);
    stats_.cost_decrease.emplace_back(dJ);
    stats_.iterations++;
  }

  /**
   * @brief Checks if the solver is done solving and can stop iterating
   *
   * The solver can exit because it has successfully converged or because it
   * has entered a bad state and needs to exit.
   *
   * @return true If the solver should stop iterating
   */
  bool IsDone() {
    bool cost_decrease = stats_.cost_decrease.back() < opts_.cost_tolerance;
    bool gradient = stats_.gradient.back() < opts_.gradient_tolerance;
    if (cost_decrease && gradient) {
      status_ = SolverStatus::kSolved;
      return true;
    }

    if (stats_.iterations >= opts_.max_iterations) {
      status_ = SolverStatus::kMaxIterations;
      return true;
    }

    if (status_ != SolverStatus::kUnsolved) {
      return true;
    }

    return false;
  }

  /**
   * @brief Initialize the solver to pre-compute any needed information and
   * be ready for a solve.
   *
   * This method should ensure the solver enters a reproducible state prior
   * to each solve, so that the `Solve()` method can be called multiple times.
   *
   */
  void Initialize() { stats_.Reset(); }
  
  /**
   * @brief Perform any operations needed to return the solver to a desireable
   * state after the iterations have stopped.
   *
   */
  void WrapUp() {}

  /**
   * @brief Calculate the infinity-norm of the feedforward gains, normalized
   * by the current control values.
   *
   * Provides an approximation to the gradient of the Lagrangian.
   *
   * @return double
   */
  double NormalizedFeedforwardGain() {
    for (int k = 0; k < N_; ++k) {
      VectorNd<m>& d = GetKnotPointFunction(k).GetFeedforwardGain();
      grad_(k) = (d.array().abs() / (Z_->Control(k).array().abs() + 1)).maxCoeff();
    }
    return grad_.sum() / grad_.size();
  }

 private:
  void Init() {
    status_ = SolverStatus::kUnsolved;
    costs_ = VectorXd::Zero(N_ + 1);
    grad_ = VectorXd::Zero(N_);
    deltaV_[0] = 0.0;
    deltaV_[1] = 0.0;
    stats_.SetCapacity(opts_.max_iterations);
    rho_ = opts_.bp_reg_initial;
    drho_ = 0.0;
  }

  /**
   * @brief Calculate the cost of each individual knot point
   *
   * @param Z
   */
  void CalcIndividualCosts(const Trajectory<n, m>& Z) {
    // TODO(bjackson): do this in parallel
    for (int k = 0; k <= N_; ++k) {
      costs_(k) = GetKnotPointFunction(k).Cost(Z.State(k), Z.Control(k));
    }
  }

  /**
   * @brief Increase the regularization, steering the steps closer towards
   * gradient descent (more robust, less efficient).
   *
   */
  void IncreaseRegularization() {
    drho_ = std::max(drho_ * opts_.bp_reg_increase_factor, opts_.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts_.bp_reg_min);
    rho_ = std::min(rho_, opts_.bp_reg_max);
  }

  /**
   * @brief Decrease the regularization term.
   *
   */
  void DecreaseRegularization() {
    drho_ = std::min(drho_ / opts_.bp_reg_increase_factor, 1 / opts_.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts_.bp_reg_min);
    rho_ = std::min(rho_, opts_.bp_reg_max);
  }

  int N_;  // number of segments
  VectorNd<n> initial_state_;
  iLQROptions opts_;
  iLQRStats stats_;  // solver statistics (iterations, cost at each iteration, etc.)
  std::vector<std::unique_ptr<KnotPointFunctions<n, m>>>
      knotpoints_;                          // problem description and data
  std::shared_ptr<Trajectory<n, m>> Z_;     // current guess for the trajectory
  std::unique_ptr<Trajectory<n, m>> Zbar_;  // temporary trajectory for forward pass

  SolverStatus status_ = SolverStatus::kUnsolved;

  VectorXd costs_;                 // costs at each knot point
  VectorXd grad_;                  // gradient at each knot point
  double rho_ = 0.0;               // regularization
  double drho_ = 0.0;              // regularization derivative (damping)
  double deltaV_[2] = {0.0, 0.0};  // terms of the expected cost decrease
};

}  // namespace ilqr
}  // namespace altro
