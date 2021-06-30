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

class KnotPointFunctionsBase {
 public:
  virtual double Cost(const Eigen::Ref<const VectorXd>& x,
                      const Eigen::Ref<const VectorXd>& u) const = 0;

  virtual void Dynamics(const Eigen::Ref<const VectorXd>& x,
                        const Eigen::Ref<const VectorXd>& u, float t, float h,
                        Eigen::Ref<VectorXd> xnext) const = 0;

  virtual void CalcCostExpansion(const Eigen::Ref<const VectorXd>& x,
                                 const Eigen::Ref<const VectorXd>& u) = 0;

  virtual void CalcDynamicsExpansion(const Eigen::Ref<const VectorXd>& x,
                                     const Eigen::Ref<const VectorXd>& u,
                                     const float t, const float h) = 0;
};

template <int n, int m>
class KnotPointFunctions : public KnotPointFunctionsBase,
                           public StateControlSized<n, m> {
  using DynamicsPtr = std::shared_ptr<problem::DiscreteDynamics>;
  using CostFunPtr = std::shared_ptr<problem::CostFunction>;
  using JacType = Eigen::Matrix<double, n, AddSizes(n, m)>;

 public:
  KnotPointFunctions(DynamicsPtr dynamics, CostFunPtr costfun)
      : StateControlSized<n, m>(
            // Use comma operator to check the dynamics pointer before using it
            // Must apply to both arguments since argument evaluation order is
            // undefined
            (ALTRO_ASSERT(dynamics != nullptr,
                          "Cannot provide a null dynamics pointer. For "
                          "terminal knot point, use KnotPointFunctions(n, m, "
                          "N)."),
             dynamics->StateDimension()),
            (ALTRO_ASSERT(dynamics != nullptr,
                          "Cannot provide a null dynamics pointer. For "
                          "terminal knot point, use KnotPointFunctions(n, m, "
                          "N)."),
             dynamics->ControlDimension())),
        model_ptr_(dynamics),
        costfun_ptr_(
            (ALTRO_ASSERT(costfun != nullptr,
                          "Cannot provide a null cost function pointer"),
             costfun)),
        cost_expansion_(dynamics->StateDimension(),
                        dynamics->ControlDimension()),
        dynamics_expansion_(dynamics->StateDimension(),
                            dynamics->ControlDimension()),
        action_value_expansion_(dynamics->StateDimension(),
                                dynamics->ControlDimension()),
        action_value_expansion_regularized_(dynamics->StateDimension(),
                                            dynamics->ControlDimension()) {
    Init();
  }
  KnotPointFunctions(int state_dim, int control_dim, CostFunPtr costfun)
      : StateControlSized<n, m>(state_dim, control_dim),
        model_ptr_(nullptr),
        costfun_ptr_(costfun),
        cost_expansion_(state_dim, control_dim),
        dynamics_expansion_(state_dim, control_dim),
        action_value_expansion_(state_dim, control_dim),
        action_value_expansion_regularized_(state_dim, control_dim) {
    Init();
  }

  double Cost(const Eigen::Ref<const VectorXd>& x,
              const Eigen::Ref<const VectorXd>& u) const {
    return costfun_ptr_->Evaluate(x, u);
  }

  void Dynamics(const Eigen::Ref<const VectorXd>& x,
                const Eigen::Ref<const VectorXd>& u, float t, float h,
                Eigen::Ref<VectorXd> xnext) const {
    model_ptr_->EvaluateInplace(x, u, t, h, xnext);
  }

  void CalcCostExpansion(const Eigen::Ref<const VectorXd>& x,
                         const Eigen::Ref<const VectorXd>& u) {
    cost_expansion_.CalcExpansion(*costfun_ptr_, x, u);
  }

  void CalcDynamicsExpansion(const Eigen::Ref<const VectorXd>& x,
                             const Eigen::Ref<const VectorXd>& u, const float t,
                             const float h) {
    if (model_ptr_) {
      dynamics_expansion_.CalcExpansion(*model_ptr_, x, u, t, h);
    }
  }

  void CalcTerminalCostToGo() {
    ctg_hessian_ = cost_expansion_.dxdx();
    ctg_gradient_ = cost_expansion_.dx();
  }

  void CalcActionValueExpansion(
      const Eigen::Ref<const MatrixXd>& ctg_hessian,
      const Eigen::Ref<const MatrixXd>& ctg_gradient) {
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

  void RegularizeActionValue(double rho,
                             BackwardPassRegularization reg_type =
                                 BackwardPassRegularization::kControlOnly) {
    action_value_expansion_regularized_ = action_value_expansion_;
    switch (reg_type) {
      case BackwardPassRegularization::kControlOnly : {
        action_value_expansion_regularized_.dudu() +=
            Eigen::Matrix<double, m, m>::Identity(this->m_, this->m_) * rho;
        break;
      }
    }
  }

  Eigen::ComputationInfo CalcGains() {
    // TODO(bjackson): Store factorization in the class
    Eigen::LLT<Eigen::Matrix<double, m, m>> Quu_chol;
    Quu_chol.compute(action_value_expansion_regularized_.dudu());
    Eigen::ComputationInfo info = Quu_chol.info();
    if (info == Eigen::Success) {
      feedback_gain_ =
          Quu_chol.solve(action_value_expansion_regularized_.dxdu().transpose());
      feedback_gain_ *= -1;
      feedforward_gain_ =
          Quu_chol.solve(action_value_expansion_regularized_.du());
      feedforward_gain_ *= -1;
    }
    return info;
  }

  void CalcCostToGo() {
    Eigen::Matrix<double, m, n>& K = GetFeedbackGain();
    Eigen::Matrix<double, m, 1>& d = GetFeedforwardGain();
    CostExpansion<n, m>& Q = GetActionValueExpansion();
    ctg_gradient_ = Q.dx() + K.transpose() * Q.dudu() * d +
                    K.transpose() * Q.du() + Q.dxdu() * d;
    ctg_hessian_ = Q.dxdx() + K.transpose() * Q.dudu() * K +
                   K.transpose() * Q.dxdu().transpose() + Q.dxdu() * K;
    ctg_delta_[0] = d.dot(Q.du());
    ctg_delta_[1] = 0.5 * d.dot(Q.dudu() * d);
  }

  void AddCostToGo(double deltaV[2]) const {
    deltaV[0] += ctg_delta_[0];
    deltaV[1] += ctg_delta_[1];
  }

  // Getters
  std::shared_ptr<problem::DiscreteDynamics> GetModelPtr() {
    return model_ptr_;
  }
  std::shared_ptr<problem::CostFunction> GetCostFunPtr() {
    return costfun_ptr_;
  }
  CostExpansion<n, m>& GetCostExpansion() { return cost_expansion_; }
  DynamicsExpansion<n, m>& GetDynamicsExpansion() {
    return dynamics_expansion_;
  }

  Eigen::Matrix<double, n, n>& GetCostToGoHessian() { return ctg_hessian_; }
  Eigen::Matrix<double, n, 1>& GetCostToGoGradient() { return ctg_gradient_; }
  double GetCostToGoDelta(const double alpha = 1.0) {
    return alpha * ctg_delta_[0] + alpha * alpha * ctg_delta_[1];
  }
  CostExpansion<n, m>& GetActionValueExpansion() {
    return action_value_expansion_;
  }
  Eigen::Matrix<double, m, n>& GetFeedbackGain() { return feedback_gain_; }
  Eigen::Matrix<double, m, 1>& GetFeedforwardGain() {
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
  double ctg_delta_[2];
};

}  // namespace ilqr
}  // namespace altro
