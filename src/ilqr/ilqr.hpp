#pragma once

#include <map>
#include <memory>

#include "problem/costfunction.hpp"
#include "problem/dynamics.hpp"
#include "eigentypes.hpp"
#include "problem/problem.hpp"
#include "ilqr/cost_expansion.hpp"

namespace altro {
namespace ilqr {

class DynamicsExpansion {};

class KnotPointFunctions {
 public:
  double Cost(const VectorXd& x, const VectorXd& u) const;
  double Dynamics(const VectorXd& x, const VectorXd& u, float t, float h) const;
  void CalcCostExpansion(const VectorXd& x, const VectorXd& u) const;
  void CalcDynamicsExpansion(const VectorXd& x, const VectorXd& u, float t,
                             float h) const;

 private:
  std::shared_ptr<problem::DiscreteDynamics> dynamics_ptr_;
  std::shared_ptr<problem::CostFunction> costfun_ptr_;

  CostExpansion<-1,-1> cost_expansion_;
  DynamicsExpansion dynamics_expansion_;
};

struct iLQROptions {
  int iterations = 100;
};

class iLQR {
 public:
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
  void Initialize();
  void Rollout();
  void UpdateExpansions();
  void BackwardPass();
  void ForwardPass();
  void EvaluateConvergence();
  bool IsConverged();
  void WrapUp();

 private:
  iLQROptions opts_;
};

}  // namespace ilqr
}  // namespace altro
