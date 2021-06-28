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
                            dynamics->ControlDimension()) {}
  KnotPointFunctions(int state_dim, int control_dim, CostFunPtr costfun) 
      : StateControlSized<n, m>(state_dim, control_dim),
        model_ptr_(nullptr),
        costfun_ptr_(costfun),
        cost_expansion_(state_dim, control_dim),
        dynamics_expansion_(state_dim, control_dim) {}

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
    dynamics_expansion_.CalcExpansion(*model_ptr_, x, u, t, h);
  }

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

 private:
  std::shared_ptr<problem::DiscreteDynamics> model_ptr_;
  std::shared_ptr<problem::CostFunction> costfun_ptr_;

  CostExpansion<n, m> cost_expansion_;
  DynamicsExpansion<n, m> dynamics_expansion_;
};

}  // namespace ilqr
}  // namespace altro
