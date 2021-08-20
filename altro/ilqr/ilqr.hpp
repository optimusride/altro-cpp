#pragma once

#include <array>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <thread>

#include "altro/common/solver_stats.hpp"
#include "altro/common/state_control_sized.hpp"
#include "altro/common/threadpool.hpp"
#include "altro/common/timer.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace ilqr {

/**
 * @brief Solve an unconstrained trajectory optimization problem using
 * iterative LQR.
 *
 * The class can be default constructed or initialized for a given number of
 * knot points. Currently, once set, the number of knot points cannot be changed.
 * If default-initialized, the number of knot points is pulled from the problem
 * in the first call to `CopyFromProblem`.
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
  explicit iLQR(int N) : N_(N), knotpoints_() { ResetInternalVariables(); }
  explicit iLQR(const problem::Problem& prob)
      : N_(prob.NumSegments()), initial_state_(std::move(prob.GetInitialStatePointer())) {
    InitializeFromProblem(prob);
  }

  iLQR(const iLQR& other) = delete;
  iLQR& operator=(const iLQR& other) = delete;
  iLQR(iLQR&& other) noexcept : N_(other.N_),
                                initial_state_(std::move(other.initial_state_)),
                                stats_(std::move(other.stats_)),
                                knotpoints_(std::move(other.knotpoints_)),
                                Z_(std::move(other.Z_)),
                                Zbar_(std::move(other.Zbar_)),
                                status_(other.status_),
                                costs_(std::move(other.costs_)),
                                grad_(std::move(other.grad_)),
                                rho_(other.rho_),
                                drho_(other.drho_),
                                deltaV_(std::move(other.deltaV_)),
                                is_initial_state_set(other.is_initial_state_set),
                                max_violation_callback_(std::move(other.max_violation_callback_)) {}

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
   * Appends the knotpoints to those currently in the solver.
   *
   * Captures the initial state from the problem as a shared pointer, so the
   * initial state of the solver is changed by modifying the initial state of
   * the original problem.
   *
   * @tparam n2 Compile-time state dimension. Can be Eigen::Dynamic (-1)
   * @tparam m2 Compile-time control dimension. Can be Eigen::Dynamic (-1)
   * @param prob Trajectory optimization problem
   * @param k_start Starting index (inclusive) for data to copy. 0 <= k_start < N+1
   * @param k_stop Terminal index (exclusive) for data to copy. 0 < k_stop <= N+1
   */
  template <int n2 = n, int m2 = m>
  void CopyFromProblem(const problem::Problem& prob, int k_start, int k_stop) {
    ALTRO_ASSERT(0 <= k_start && k_start <= N_,
                 fmt::format("Start index must be in the interval [0,{}]", N_));
    ALTRO_ASSERT(0 <= k_stop && k_stop <= N_ + 1,
                 fmt::format("Start index must be in the interval [0,{}]", N_ + 1));
    ALTRO_ASSERT(prob.IsFullyDefined(), "Expected problem to be fully defined.");
    for (int k = k_start; k < k_stop; ++k) {
      if (n != Eigen::Dynamic) {
        ALTRO_ASSERT(
            prob.GetDynamics(k)->StateDimension() == n,
            fmt::format("Inconsistent state dimension at knot point {}. Expected {}, got {}", k, n,
                        prob.GetDynamics(k)->StateDimension()));
      }
      if (m != Eigen::Dynamic) {
        ALTRO_ASSERT(
            prob.GetDynamics(k)->ControlDimension() == m,
            fmt::format("Inconsistent control dimension at knot point {}. Expected {}, got {}", k,
                        m, prob.GetDynamics(k)->ControlDimension()));
      }
      std::shared_ptr<problem::DiscreteDynamics> model = prob.GetDynamics(k);
      std::shared_ptr<problem::CostFunction> costfun = prob.GetCostFunction(k);
      knotpoints_.emplace_back(std::make_unique<ilqr::KnotPointFunctions<n2, m2>>(model, costfun));
    }
    initial_state_ = prob.GetInitialStatePointer();
    is_initial_state_set = true;
  }

  template <int n2 = n, int m2 = m>
  void InitializeFromProblem(const problem::Problem& prob) {
    ALTRO_ASSERT(prob.NumSegments() == N_,
                 fmt::format("Number of segments in problem {}, should be equal to the number of "
                             "segments in the solver, {}",
                             prob.NumSegments(), N_));
    CopyFromProblem<n2, m2>(prob, 0, N_ + 1);
    ResetInternalVariables();
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

  SolverStats& GetStats() { return stats_; }
  const SolverStats& GetStats() const { return stats_; }
  SolverOptions& GetOptions() { return stats_.GetOptions(); }
  const SolverOptions& GetOptions() const { return stats_.GetOptions(); }
  VectorXd& GetCosts() { return costs_; }
  SolverStatus GetStatus() const { return status_; }
  std::shared_ptr<VectorXd> GetInitialState() { return initial_state_; }
  double GetRegularization() { return rho_; }

  /**
   * @brief Get the assignment of the trajectory into tasks.
   *
   * A task is defined by a set of consecutive knot point indices whose
   * expansions will be processed serially. Although all knot points can be
   * processed in parallel, it's usually better to "chunk" the trajectory into
   * the number of available parallel processors.
   *
   * Most users will not need to consume this information.
   *
   * @return std::vector<int>& A vector of strictly increasing knot point indices.
   * Each tasks processes knotpoints in the interval [`inds[k]`, `inds[k+1]`),
   * where `inds[0] = 0` and inds.back() = N+1`.
   *
   */
  std::vector<int>& GetTaskAssignment() {
    if (ShouldRedoTaskAssignment()) {
      DefaultTaskAssignment();
    }
    return work_inds_;
  }

  /**
   * @brief Get the number of threads used in the iLQR solver
   *
   * @return Number of threads
   */
  size_t NumThreads() const { return pool_.NumThreads(); }

  /**
   * @brief Get the number of tasks that can be executed in parallel.
   *
   * Controlled via `AssignWork`.
   *
   */
  int NumTasks() const { return work_inds_.size() - 1; }

  /**
   * @brief Create a new zero-initialized trajectory.
   *
   * Assumes a uniform time step.
   * The trajectory is automatically linked to the solver and is used
   * both as the initial guess and as the storage location for the optimized
   * solution during and after the solve.
   *
   * @param dt Time step used in the trajectory.
   * @return std::shared_ptr<Trajectory<n, m>> A new zero-initialized trajectory.
   */
  std::shared_ptr<Trajectory<n, m>> MakeTrajectory(float dt) {
    Z_ = std::make_shared<Trajectory<n, m>>(NumSegments());
    Z_->SetUniformStep(dt);
    return Z_;
  }

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

  void SetConstraintCallback(const std::function<double()>& max_violation) {
    max_violation_callback_ = max_violation;
  }

  /**
   * @brief Set the division of knot points indices into parallelizable tasks.
   *
   * Defines groups of consecutive knotpoints that should be processed in series
   * as a single task. Each group can then be run independently and in parallel.
   * For best performance, the number of tasks should be equal to the number of
   * available cores.
   *
   * Once this is set, the solver will no longer automatically adjust the
   * number of tasks if the number of requested threads (via `GetOptions().nthreads`)
   * or tasks per thread (via `GetOptions().tasks_per_thread`) changes.
   * changes. Is is the user's responsibility to modify this as needed once set.
   *
   * @param inds A strictly increasing vector of knot point indices. For a vector
   * of length N, it defines N-1 tasks, where each task processes indices in the
   * interval [`inds[i]`, `inds[i+1]`).
   */
  void SetTaskAssignment(std::vector<int> inds) {
    ALTRO_ASSERT(work_inds_.back() == NumSegments() + 1,
                 "Work inds should include the terminal index.");
    ALTRO_ASSERT(work_inds_[0] == 0, "Work inds should start with a 0.");
    ALTRO_ASSERT(work_inds_.size() >= 2, "Work inds must have at least 2 elements.");
    bool is_sorted = true;
    for (int i = 1; i < inds.size(); ++i) {
      if (inds[i] <= inds[i - 1]) {
        is_sorted = false;
      }
    }
    ALTRO_ASSERT(is_sorted, "Work inds must be a set of strictly increasing integers.");
    work_inds_ = std::move(inds);
    custom_work_assignment_ = true;
  }

  /***************************** Algorithm **************************************/
  /**
   * @brief Solve the trajectory optimization problem using iLQR
   *
   * @post The provided trajectory is overwritten with a locally-optimal
   * dynamically-feasible trajectory. The solver status and statistics,
   * obtained via GetStatus() and GetStats() are updated.
   * The solve is successful if `GetStatus == SolverStatus::kSuccess`.
   *
   */
  void Solve() {
    ALTRO_ASSERT(is_initial_state_set, "Initial state must be set before solving.");
    ALTRO_ASSERT(Z_ != nullptr, "Invalid trajectory pointer. May be uninitialized.");

    // TODO(bjackson): Allow the solver to optimize a portion of a longer trajectory?
    ALTRO_ASSERT(Z_->NumSegments() == N_,
                 fmt::format("Initial trajectory must have length {}", N_));

    // Start profiler
    GetOptions().profiler_enable ? stats_.GetTimer()->Activate() : stats_.GetTimer()->Deactivate();
    Stopwatch sw = stats_.GetTimer()->Start("ilqr");

    SolveSetup();  // reset any internal variables
    Rollout();     // simulate the system forward using initial controls
    stats_.initial_cost = Cost();

    for (int iter = 0; iter < GetOptions().max_iterations_inner; ++iter) {
      UpdateExpansions();
      BackwardPass();
      ForwardPass();
      UpdateConvergenceStatistics();

      if (stats_.GetVerbosity() >= LogLevel::kInner) {
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
  double Cost() {
    ALTRO_ASSERT(Z_ != nullptr, "Invalid trajectory pointer. May be uninitialized.");
    return Cost(*Z_);
  }
  double Cost(const Trajectory<n, m>& Z) {
    Stopwatch sw = stats_.GetTimer()->Start("cost");
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
    Stopwatch sw = stats_.GetTimer()->Start("expansions");
    ALTRO_ASSERT(Z_ != nullptr, "Trajectory pointer must be set before updating the expansions.");

    int nthreads = NumThreads();
    if (nthreads <= 1) {
      UpdateExpansionsBlock(0, NumSegments() + 1);
    } else {
      {
        Stopwatch sw2 = stats_.GetTimer()->Start("add_tasks");
        for (const std::function<void()>& task : tasks_) {
          pool_.AddTask(task);
        }
      }
      pool_.Wait();
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
    Stopwatch sw = stats_.GetTimer()->Start("backward_pass");

    // Regularization
    Eigen::ComputationInfo info;

    // Terminal Cost-to-go
    knotpoints_[N_]->CalcTerminalCostToGo();
    Eigen::Matrix<double, n, n>* Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
    Eigen::Matrix<double, n, 1>* Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

    int max_reg_count = 0;
    deltaV_[0] = 0.0;
    deltaV_[1] = 0.0;

    bool repeat_backwardpass = true;
    while (repeat_backwardpass) {
      for (int k = N_ - 1; k >= 0; --k) {
        // TODO(bjackson)[SW-16103] Create a test that checks this
        knotpoints_[k]->CalcActionValueExpansion(*Sxx_prev, *Sx_prev);
        knotpoints_[k]->RegularizeActionValue(rho_);
        info = knotpoints_[k]->CalcGains();

        // Handle solve failure
        if (info != Eigen::Success) {

          IncreaseRegularization();

          // Reset the cost-to-go pointers to the terminal expansion
          Sxx_prev = &(knotpoints_[N_]->GetCostToGoHessian());
          Sx_prev = &(knotpoints_[N_]->GetCostToGoGradient());

          // Check if we're at max regularization
          if (rho_ >= GetOptions().bp_reg_max) {
            max_reg_count++;
          }

          if (max_reg_count >= GetOptions().bp_reg_fail_threshold) {
            status_ = SolverStatus::kBackwardPassRegularizationFailed;
            repeat_backwardpass = false;
          }
          break;
        }

        // Update Cost-To-Go
        knotpoints_[k]->CalcCostToGo();
        knotpoints_[k]->AddCostToGo(&deltaV_);

        Sxx_prev = &(knotpoints_[k]->GetCostToGoHessian());
        Sx_prev = &(knotpoints_[k]->GetCostToGoGradient());

        // Backward pass successful if it calculates the cost to go at
        // the first knot point.
        if (k == 0) {
          repeat_backwardpass = false;
        }
      } // end for
    } // end while
    stats_.Log("reg", rho_);
    DecreaseRegularization();
  }

  /**
   * @brief Simulate the dynamics forward from the initial state
   *
   * By default it will simulate the system forward open-loop.
   *
   */
  void Rollout() {
    Z_->State(0) = *initial_state_;
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
    Stopwatch sw = stats_.GetTimer()->Start("rollout");

    Zbar_->State(0) = *initial_state_;
    for (int k = 0; k < N_; ++k) {
      MatrixNxMd<m, n>& K = GetKnotPointFunction(k).GetFeedbackGain();
      VectorNd<m>& d = GetKnotPointFunction(k).GetFeedforwardGain();

      // TODO(bjackson): Make this a function of the dynamics
      VectorNd<n> dx = Zbar_->State(k) - Z_->State(k);
      Zbar_->Control(k) = Z_->Control(k) + K * dx + d * alpha;

      // Simulate forward with feedback
      GetKnotPointFunction(k).Dynamics(Zbar_->State(k), Zbar_->Control(k), Zbar_->GetTime(k),
                                       Zbar_->GetStep(k), Zbar_->State(k + 1));

      if (GetOptions().check_forwardpass_bounds) {
        if (Zbar_->State(k + 1).norm() > GetOptions().state_max) {
          // TODO(bjackson): Emit warning (need logging mechanism)
          status_ = SolverStatus::kStateLimit;
          return false;
        }
        if (Zbar_->Control(k).norm() > GetOptions().control_max) {
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
    Stopwatch sw = stats_.GetTimer()->Start("forward_pass");
    SolverOptions& opts = GetOptions();

    double J0 = costs_.sum();  // Calculated during UpdateExpansions

    double alpha = 1.0;
    double z = -1.0;
    int iter_fp = 0;
    bool success = false;

    double J = J0;

    for (; iter_fp < opts.line_search_max_iterations; ++iter_fp) {
      if (RolloutClosedLoop(alpha)) {
        J = Cost(*Zbar_);
        double expected = -alpha * (deltaV_[0] + alpha * deltaV_[1]);
        if (expected > 0.0) {
          z = (J0 - J) / expected;
        } else {
          z = -1.0;
        }

        if (opts.line_search_lower_bound <= z && z <= opts.line_search_upper_bound && J < J0) {
          success = true;
          // stats_.improvement_ratio.emplace_back(z);
          stats_.Log("cost", J);
          stats_.Log("alpha", alpha);
          stats_.Log("z", z);
          break;
        }
      }
      alpha /= opts.line_search_decrease_factor;
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
    Stopwatch sw = stats_.GetTimer()->Start("stats");

    double dgrad = NormalizedFeedforwardGain();
    double dJ = 0.0;
    if (stats_.iterations_inner == 0) {
      dJ = stats_.initial_cost - stats_.cost.back();
    } else {
      dJ = stats_.cost.rbegin()[1] - stats_.cost.rbegin()[0];
    }

    // stats_.gradient.emplace_back(dgrad);
    stats_.iterations_inner++;
    stats_.iterations_total++;
    stats_.Log("dJ", dJ);
    stats_.Log("viol", max_violation_callback_());
    stats_.Log("iters", stats_.iterations_total);
    stats_.Log("grad", dgrad);
    stats_.NewIteration();
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
    Stopwatch sw = stats_.GetTimer()->Start("convergence_check");
    SolverOptions& opts = GetOptions();

    bool cost_decrease = stats_.cost_decrease.back() < opts.cost_tolerance;
    bool gradient = stats_.gradient.back() < opts.gradient_tolerance;
    bool is_done = false;

    if (cost_decrease && gradient) {
      status_ = SolverStatus::kSolved;
      is_done = true;
    } else if (stats_.iterations_inner >= opts.max_iterations_inner) {
      status_ = SolverStatus::kMaxInnerIterations;
      is_done = true;
    } else if (stats_.iterations_total >= opts.max_iterations_total) {
      status_ = SolverStatus::kMaxIterations;
      is_done = true;
    } else if (status_ != SolverStatus::kUnsolved) {
      is_done = true;
    }

    return is_done;
  }

  /**
   * @brief Initialize the solver to pre-compute any needed information and
   * be ready for a solve.
   *
   * This method should ensure the solver enters a reproducible state prior
   * to each solve, so that the `Solve()` method can be called multiple times.
   *
   */
  void SolveSetup() {
    Stopwatch sw = stats_.GetTimer()->Start("init");
    stats_.iterations_inner = 0;
    stats_.SetVerbosity(GetOptions().verbose);

    // Make sure Zbar has the same times as the initial trajectory
    if (Z_ != nullptr) {
      int k;
      for (k = 0; k < N_; ++k) {
        Zbar_->SetStep(k, Z_->GetStep(k));
        Zbar_->SetTime(k, Z_->GetTime(k));
      }
      Zbar_->SetTime(N_, Z_->GetTime(N_));
    }

    ResetInternalVariables();
  }

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

  void UpdateExpansionsBlock(int start, int stop) {
    for (int k = start; k < stop; ++k) {
      KnotPoint<n, m>& z = Z_->GetKnotPoint(k);
      knotpoints_[k]->CalcCostExpansion(z.State(), z.Control());
      knotpoints_[k]->CalcDynamicsExpansion(z.State(), z.Control(), z.GetTime(), z.GetStep());
      costs_(k) = GetKnotPointFunction(k).Cost(z.State(), z.Control());
    }
  }

 private:
  void ResetInternalVariables() {
    status_ = SolverStatus::kUnsolved;
    costs_ = VectorXd::Zero(N_ + 1);
    grad_ = VectorXd::Zero(N_);
    deltaV_[0] = 0.0;
    deltaV_[1] = 0.0;
    rho_ = GetOptions().bp_reg_initial;
    drho_ = 0.0;

    LaunchThreads();
  }

  /**
   * @brief Check if the tasks need to re-assigned.
   *
   * Will not overrite tasks once they have been assigned manually via
   * `SetTaskAssignment()`.
   *
   */
  bool ShouldRedoTaskAssignment() const {
    bool is_custom = custom_work_assignment_;
    int tasks_per_thread = GetOptions().tasks_per_thread;
    bool has_expected_number_of_tasks = NumTasks() == GetOptions().NumThreads() * tasks_per_thread;
    bool keep_assignment = is_custom || has_expected_number_of_tasks;
    return !keep_assignment;
  }

  void LaunchThreads() {
    size_t nthreads = GetOptions().NumThreads();

    // Reset the thread pool if the requested number of threads changes
    int threadpool_size = NumThreads();
    bool single_threaded = threadpool_size == 0 && nthreads == 1;
    if (single_threaded) {
      return;
    }
    bool num_threads_changed = nthreads != NumThreads();
    if (N_ > 0 && (num_threads_changed || ShouldRedoTaskAssignment())) {
      if (pool_.IsRunning()) {
        pool_.StopThreads();
      }
      tasks_.clear();

      // Create tasks
      std::vector<int>& work_inds = GetTaskAssignment();
      int ntasks = NumTasks();
      for (int i = 0; i < ntasks; ++i) {
        int start = work_inds[i];
        int stop = work_inds[i + 1];
        auto expansion_block = [this, start, stop]() { UpdateExpansionsBlock(start, stop); };
        tasks_.emplace_back(std::move(expansion_block));
      }

      // Start the pool
      if (nthreads > 1) {
        pool_.LaunchThreads(nthreads);
      }
    }
  }

  void DefaultTaskAssignment() {
    int nthreads = GetOptions().NumThreads();
    int ntasks = nthreads * GetOptions().tasks_per_thread;
    double step = NumSegments() / static_cast<double>(ntasks);
    work_inds_.clear();
    for (double val = 0.0; val <= NumSegments(); val += step) {
      work_inds_.emplace_back(static_cast<int>(round(val)));
    }
    ALTRO_ASSERT(work_inds_.back() == NumSegments(),
                 "Work inds should include the terminal index.");
    work_inds_.back() += 1;  // Increment the last index to include the terminal index.
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
    const SolverOptions& opts = GetOptions();
    drho_ = std::max(drho_ * opts.bp_reg_increase_factor, opts.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts.bp_reg_min);
    rho_ = std::min(rho_, opts.bp_reg_max);
  }

  /**
   * @brief Decrease the regularization term.
   *
   */
  void DecreaseRegularization() {
    const SolverOptions& opts = GetOptions();
    drho_ = std::min(drho_ / opts.bp_reg_increase_factor, 1 / opts.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts.bp_reg_min);
    rho_ = std::min(rho_, opts.bp_reg_max);
  }

  int N_;  // number of segments
  std::shared_ptr<VectorXd> initial_state_;
  SolverStats stats_;  // solver statistics (iterations, cost at each iteration, etc.)

  // TODO(bjackson): Create a non-templated base class to allow different dimensions.
  std::vector<std::unique_ptr<KnotPointFunctions<n, m>>>
      knotpoints_;                          // problem description and data
  std::shared_ptr<Trajectory<n, m>> Z_;     // current guess for the trajectory
  std::unique_ptr<Trajectory<n, m>> Zbar_;  // temporary trajectory for forward pass

  SolverStatus status_ = SolverStatus::kUnsolved;

  VectorXd costs_;     // costs at each knot point
  VectorXd grad_;      // gradient at each knot point
  double rho_ = 0.0;   // regularization
  double drho_ = 0.0;  // regularization derivative (damping)
  std::array<double, 2> deltaV_;

  bool is_initial_state_set = false;
  bool custom_work_assignment_ = false;  // Has user assigned a custom task assignment

  std::function<double()> max_violation_callback_ = []() { return 0.0; };
  std::vector<std::function<void()>> tasks_;
  std::vector<int> work_inds_;
  ThreadPool pool_;
};

}  // namespace ilqr
}  // namespace altro
