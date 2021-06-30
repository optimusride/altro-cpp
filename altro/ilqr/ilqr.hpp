#pragma once

#include <map>

#include "altro/common/state_control_sized.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"

namespace altro {
namespace ilqr {

struct iLQROptions {
  int iterations = 100;
  double bp_reg_increase_factor = 1.6;
  double bp_reg_enable = true;
  double bp_reg_initial = 0.0;
  double bp_reg_max = 1e8;
  double bp_reg_min = 1e-8;
  double bp_reg_forwardpass = 10.0;
  int bp_reg_fail_threshold = 100;
};

template <int n = Eigen::Dynamic, int m = Eigen::Dynamic>
class iLQR {
 public:
  explicit iLQR(int N) : N_(N), opts_(), knotpoints_() {}

  explicit iLQR(const problem::Problem& prob, const iLQROptions& opts = iLQROptions()) 
      : N_(prob.NumSegments()), opts_(opts) {
    CopyFromProblem(prob, 0, N_ + 1);
  }

  /**
   * @brief Copy the data from a Problem class into the iLQR solver
   * 
   * Capture shared pointers to the cost and dynamics objects for each 
   * knot point, storing them in the correspoding KnotPointFunctions object.
   * 
   * Assumes both the problem and the solver have the same number of knot points.
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
    ALTRO_ASSERT(prob.IsFullyDefined(),
                 "Expected problem to be fully defined.");
    int state_dim = 0;
    int control_dim = 0;
    for (int k = k_start; k < k_stop; ++k) {
      std::shared_ptr<problem::DiscreteDynamics> model = prob.GetDynamics(k);
      std::shared_ptr<problem::CostFunction> costfun = prob.GetCostFunction(k);

      // Model will be nullptr at the last knot point
      if (model) {
        state_dim = model->StateDimension();
        control_dim = model->ControlDimension();
        knotpoints_.emplace_back(
            std::make_unique<ilqr::KnotPointFunctions<n2, m2>>(model, costfun));
      } else {
        // To construct the KPF at the terminal knot point we need to tell 
        // it the state and control dimensions since we don't have a dynamics
        // function
        ALTRO_ASSERT(k == N_,
                     "Expected model to only be a nullptr at last time step");
        ALTRO_ASSERT(state_dim != 0 && control_dim != 0,
                     "The last time step cannot be copied in isolation. "
                     "Include the previous time step, e.g. "
                     "CopyFromProblem(prob,N-1,N+1)");
        knotpoints_.emplace_back(std::make_unique<ilqr::KnotPointFunctions<n, m>>(
            state_dim, control_dim, costfun));
      }
    }
  }

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
  }

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

  // Just a prototype for now...
  void Solve() {
    Initialize();  // reset any internal variables
    Rollout();     // simulate the system forward using initial controls

    for (int iter = 0; iter < opts_.iterations; ++iter) {
      UpdateExpansions();  // update dynamics and cost expansion, cost
                           // (parallel)
      BackwardPass();
      ForwardPass();
      EvaluateConvergence();
      if (IsConverged()) {
        break;
      }
    }

    WrapUp();
  }

  /**
   * @brief Update the cost and dynamics expansions
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
    ALTRO_ASSERT(
        Z_ != nullptr,
        "Trajectory pointer must be set before updating the expansions.");
    for (int k = 0; k <= N_; ++k) {
      KnotPoint<n, m>& z = Z_->GetKnotPoint(k); 
      knotpoints_[k]->CalcCostExpansion(z.State(), z.Control());
      knotpoints_[k]->CalcDynamicsExpansion(z.State(), z.Control(), z.GetTime(),
                                            z.GetStep());
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
   * @param stop_index 
   */
  void BackwardPass(int stop_index = 0) {
    // Regularization
    Eigen::ComputationInfo info;

    // Terminal Cost-to-go
    knotpoints_[N_]->CalcTerminalCostToGo();
    Eigen::Matrix<double, n, n>* Sxx_prev =
        &(knotpoints_[N_]->GetCostToGoHessian());
    Eigen::Matrix<double, n, 1>* Sx_prev =
        &(knotpoints_[N_]->GetCostToGoGradient());

    int max_reg_count = 0;
    for (int k = N_ - 1; k >= stop_index; --k) {
      knotpoints_[k]->CalcActionValueExpansion(*Sxx_prev, *Sx_prev);
      knotpoints_[k]->RegularizeActionValue(rho_);
      info = knotpoints_[k]->CalcGains();

      // Handle solve failure
      if (info != Eigen::Success) {
        IncreaseRegularization();
        k = N_ - 1;  // Start at the beginning of the trajectory again

        // Check if we're at max regularization
        if (rho_ >= opts_.bp_reg_max) {
          max_reg_count++;
        }

        // Throw an error if we keep failing, even at max regularization
        // TODO(bjackson): Look at better ways of doing this
        if (max_reg_count >= opts_.bp_reg_fail_threshold) {
          throw std::runtime_error(
              "Backward pass regularization increased too many times.");
        }
        continue;
      }

      // Update Cost-To-Go
      knotpoints_[k]->CalcCostToGo();
      knotpoints_[k]->AddCostToGo(deltaV_);

      Sxx_prev = &(knotpoints_[k]->GetCostToGoHessian());
      Sx_prev = &(knotpoints_[k]->GetCostToGoGradient());
    }
  }

  void Initialize();
  void Rollout();
  void ForwardPass();
  void EvaluateConvergence();
  bool IsConverged();
  void WrapUp();

 private:
  void IncreaseRegularization() {
    drho_ = std::max(drho_ * opts_.bp_reg_increase_factor,
                     opts_.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts_.bp_reg_min);
    rho_ = std::min(rho_, opts_.bp_reg_max);
  }
  void DecreaseRegularization() {
    drho_ = std::min(drho_ / opts_.bp_reg_increase_factor,
                     1 / opts_.bp_reg_increase_factor);
    rho_ = std::max(rho_ * drho_, opts_.bp_reg_min);
    rho_ = std::min(rho_, opts_.bp_reg_max);
  }

  int N_;
  iLQROptions opts_;
  std::vector<std::unique_ptr<KnotPointFunctions<n, m>>> knotpoints_;
  std::shared_ptr<Trajectory<n, m>> Z_;

  double rho_;
  double drho_;
  double deltaV_[2];
};

}  // namespace ilqr
}  // namespace altro
