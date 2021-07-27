#pragma once

#include <memory>

#include "altro/common/state_control_sized.hpp"
#include "altro/eigentypes.hpp"
#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/dynamics_expansion.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/problem/dynamics.hpp"

namespace altro {
namespace ilqr {

// TODO(bjackson): implement other regularization methods
enum class BackwardPassRegularization {
  kControlOnly,
  // kStateOnly,
  // kStateControl
};

/**
 * @brief Stores the methods and data to evaluate various expressions at
 * each knot point
 *
 * Stores the cost and dynamics definitions, and provides methods to evaluate
 * their expansions. Also provides methods to calculate terms needed by the
 * iLQR backward pass, and stores the action-value expansion, the quadratic
 * approximation of the cost-to-go, and the feedback and feedforward gains.
 *
 * @tparam n Compile-time state dimension
 * @tparam m Compile-time control dimension
 */
template <int n, int m>
class KnotPointFunctions : public StateControlSized<n, m> {
  using DynamicsPtr = std::shared_ptr<problem::DiscreteDynamics>;
  using CostFunPtr = std::shared_ptr<problem::CostFunction>;
  using JacType = Eigen::Matrix<double, n, AddSizes(n, m)>;

public:
  KnotPointFunctions(DynamicsPtr dynamics, CostFunPtr costfun)
      : StateControlSized<n, m>(
            // Use comma operator to check the dynamics pointer before using it.
            // Must apply to both arguments since argument evaluation order is undefined.
            (CheckDynamicsPtr(dynamics), dynamics->StateDimension()),
            (CheckDynamicsPtr(dynamics), dynamics->ControlDimension())),
        model_ptr_(std::move(dynamics)),
        costfun_ptr_(std::move(costfun)),
        cost_expansion_(model_ptr_->StateDimension(),
                        model_ptr_->ControlDimension()),
        dynamics_expansion_(model_ptr_->StateDimension(),
                            model_ptr_->ControlDimension()),
        action_value_expansion_(model_ptr_->StateDimension(),
                                model_ptr_->ControlDimension()),
        action_value_expansion_regularized_(model_ptr_->StateDimension(),
                                            model_ptr_->ControlDimension()) {
    ALTRO_ASSERT(costfun_ptr_ != nullptr, "Cannot provide a null cost function pointer.");
    Init();
  }
  // Create the kpf for the last knot point
  KnotPointFunctions(int state_dim, int control_dim, CostFunPtr costfun)
      : StateControlSized<n, m>(state_dim, control_dim), model_ptr_(nullptr),
        costfun_ptr_(std::move(costfun)), cost_expansion_(state_dim, control_dim),
        dynamics_expansion_(state_dim, control_dim),
        action_value_expansion_(state_dim, control_dim),
        action_value_expansion_regularized_(state_dim, control_dim) {
    ALTRO_ASSERT(costfun_ptr_ != nullptr, "Cannot provide a null cost function pointer.");
    Init();
  }

  /**
   * @brief Evaluate the cost for the knot point
   *
   * @param x state vector
   * @param u control vector
   * @return double
   */
  double Cost(const VectorXdRef &x,
              const VectorXdRef &u) const {
    return costfun_ptr_->Evaluate(x, u);
  }

  /**
   * @brief Evaluate the discrete dynamics at the knot point
   *
   * @param x state vector
   * @param u control vector
   * @param t independent variable (e.g. time)
   * @param h step in independent variable
   * @param xnext states at the next knot point
   */
  void Dynamics(const VectorXdRef &x,
                const VectorXdRef &u, float t, float h,
                Eigen::Ref<VectorXd> xnext) const { // NOLINT(performance-unnecessary-value-param)
    model_ptr_->EvaluateInplace(x, u, t, h, xnext);
  }

  /**
   * @brief Evaluate the 2nd order expansion of the cost function
   *
   * @param x state vector
   * @param u control vector
   */
  void CalcCostExpansion(const VectorXdRef &x,
                         const VectorXdRef &u) {
    cost_expansion_.SetZero();
    cost_expansion_.CalcExpansion(*costfun_ptr_, x, u);
  }

  /**
   * @brief Evaluate the first-order expansion of the dynamics
   *
   * @param x state vector
   * @param u control vector
   * @param t independent variable (e.g. time)
   * @param h step in independent variable
   */
  void CalcDynamicsExpansion(const VectorXdRef &x,
                             const VectorXdRef &u, const float t,
                             const float h) {
    if (model_ptr_) {
      dynamics_expansion_.SetZero();
      dynamics_expansion_.CalcExpansion(*model_ptr_, x, u, t, h);
    }
  }

  /**
   * @brief Calculate the terminal cost-to-go, or the cost-to-go at the last
   * knot point.
   *
   */
  void CalcTerminalCostToGo() {
    ctg_hessian_ = cost_expansion_.dxdx();
    ctg_gradient_ = cost_expansion_.dx();
  }

  /**
   * @brief Calculate the action-value expansion given the quadratic
   * approximation of the cost-to-go at the next time step.
   *
   * @pre The cost and dynamics expansions must be calculated.
   *
   * @param ctg_hessian Hessian of the cost-to-go at the next time step.
   * @param ctg_gradient Gradient of the cost-to-go at the next time step.
   */
  void
  CalcActionValueExpansion(const Eigen::Ref<const MatrixXd> &ctg_hessian,
                           const Eigen::Ref<const MatrixXd> &ctg_gradient) {
    Eigen::Block<JacType, n, n> A = dynamics_expansion_.GetA();
    Eigen::Block<JacType, n, m> B = dynamics_expansion_.GetB();
    action_value_expansion_.dxdx() =
        cost_expansion_.dxdx() + A.transpose() * ctg_hessian * A;
    action_value_expansion_.dxdu() =
        cost_expansion_.dxdu() + A.transpose() * ctg_hessian * B;
    action_value_expansion_.dudu() =
        cost_expansion_.dudu() + B.transpose() * ctg_hessian * B;
    action_value_expansion_.dx() =
        cost_expansion_.dx() + A.transpose() * ctg_gradient;
    action_value_expansion_.du() =
        cost_expansion_.du() + B.transpose() * ctg_gradient;
  }

  /**
   * @brief Add regularization to the action-value expansion prior to solving
   * for the optimal feedback policy.
   *
   * @pre The action value expansion must be calculated.
   *
   * @param rho Amount of regularization
   * @param reg_type How to incorporate the regularization.
   */
  void RegularizeActionValue(const double rho,
                             BackwardPassRegularization reg_type =
                                 BackwardPassRegularization::kControlOnly) {
    action_value_expansion_regularized_ = action_value_expansion_;
    switch (reg_type) {
    case BackwardPassRegularization::kControlOnly: {
      action_value_expansion_regularized_.dudu() +=
          Eigen::Matrix<double, m, m>::Identity(this->m_, this->m_) * rho;
      break;
    }
    }
  }

  /**
   * @brief Calculate the feedback and feedforward gains by inverting the
   * Hessian of the action-value expansion with respect to u using a
   * Cholesky decomposition.
   *
   * @pre The regularized action-value expansion must be calculated.
   *
   * @return Eigen enum describing the result of the Cholesky factorization.
   */
  Eigen::ComputationInfo CalcGains() {
    // TODO(bjackson): Store factorization in the class
    Eigen::LLT<Eigen::Matrix<double, m, m>> Quu_chol;
    Quu_chol.compute(action_value_expansion_regularized_.dudu());
    Eigen::ComputationInfo info = Quu_chol.info();
    if (info == Eigen::Success) {
      feedback_gain_ = Quu_chol.solve(
          action_value_expansion_regularized_.dxdu().transpose());
      feedback_gain_ *= -1;
      feedforward_gain_ =
          Quu_chol.solve(action_value_expansion_regularized_.du());
      feedforward_gain_ *= -1;
    }
    return info;
  }

  /**
   * @brief Calculate the current quadratic approximation of the cost-to-go
   * given the feedback policy.
   *
   * @pre The gains and action-value expansion must be computed.
   *
   */
  void CalcCostToGo() {
    Eigen::Matrix<double, m, n> &K = GetFeedbackGain();
    Eigen::Matrix<double, m, 1> &d = GetFeedforwardGain();
    CostExpansion<n, m> &Q = GetActionValueExpansion();
    ctg_gradient_ = Q.dx() + K.transpose() * Q.dudu() * d +
                    K.transpose() * Q.du() + Q.dxdu() * d;
    ctg_hessian_ = Q.dxdx() + K.transpose() * Q.dudu() * K +
                   K.transpose() * Q.dxdu().transpose() + Q.dxdu() * K;
    ctg_delta_[0] = d.dot(Q.du());
    ctg_delta_[1] = 0.5 * d.dot(Q.dudu() * d); // NOLINT(readability-magic-numbers)
  }

  void AddCostToGo(std::array<double, 2>* const deltaV) const {
    (*deltaV)[0] += ctg_delta_[0];
    (*deltaV)[1] += ctg_delta_[1];
  }

  void AddCostToGo(double* const deltaV) const {
    deltaV[0] += ctg_delta_[0];
    deltaV[1] += ctg_delta_[1];
  }

  /**************************** Getters ***************************************/
  std::shared_ptr<problem::DiscreteDynamics> GetModelPtr() {
    return model_ptr_;
  }
  std::shared_ptr<problem::CostFunction> GetCostFunPtr() {
    return costfun_ptr_;
  }
  CostExpansion<n, m> &GetCostExpansion() { return cost_expansion_; }
  DynamicsExpansion<n, m> &GetDynamicsExpansion() {
    return dynamics_expansion_;
  }

  Eigen::Matrix<double, n, n> &GetCostToGoHessian() { return ctg_hessian_; }
  Eigen::Matrix<double, n, 1> &GetCostToGoGradient() { return ctg_gradient_; }
  double GetCostToGoDelta(const double alpha = 1.0) {
    return alpha * ctg_delta_[0] + alpha * alpha * ctg_delta_[1];
  }
  CostExpansion<n, m> &GetActionValueExpansion() {
    return action_value_expansion_;
  }
  CostExpansion<n, m> &GetActionValueExpansionRegularized() {
    return action_value_expansion_regularized_;
  }
  Eigen::Matrix<double, m, n> &GetFeedbackGain() { return feedback_gain_; }
  Eigen::Matrix<double, m, 1> &GetFeedforwardGain() {
    return feedforward_gain_;
  }

private:
  void Init() {
    feedback_gain_ = Eigen::Matrix<double, m, n>::Zero(this->m_, this->n_);
    feedforward_gain_ = Eigen::Matrix<double, m, 1>::Zero(this->m_, 1);
    ctg_hessian_ = Eigen::Matrix<double, n, n>::Zero(this->n_, this->n_);
    ctg_gradient_ = Eigen::Matrix<double, n, 1>::Zero(this->n_, 1);
    ctg_delta_[0] = 0;
    ctg_delta_[1] = 0;
  }

  void CheckDynamicsPtr(const DynamicsPtr& dynamics) {
    (void) dynamics;  // needed to surpress erroneous unused variable warning
    ALTRO_ASSERT(dynamics != nullptr, "Cannot provide a null dynamics pointer.");
  }

  std::shared_ptr<problem::DiscreteDynamics> model_ptr_;
  std::shared_ptr<problem::CostFunction> costfun_ptr_;

  CostExpansion<n, m> cost_expansion_;
  DynamicsExpansion<n, m> dynamics_expansion_;
  CostExpansion<n, m> action_value_expansion_;
  CostExpansion<n, m> action_value_expansion_regularized_;

  Eigen::Matrix<double, m, n> feedback_gain_;
  Eigen::Matrix<double, m, 1> feedforward_gain_;

  Eigen::Matrix<double, n, n> ctg_hessian_;
  Eigen::Matrix<double, n, 1> ctg_gradient_;
  std::array<double, 2> ctg_delta_;
};

} // namespace ilqr
} // namespace altro
