#pragma once

#include <map>

#include "altro/eigentypes.hpp"
#include "altro/ilqr/knot_point_function_type.hpp"
#include "altro/problem/problem.hpp"

namespace altro {
namespace ilqr {

struct iLQROptions {
  int iterations = 100;
};

class iLQR {
 public:
  explicit iLQR(int N) : N_(N), opts_(), knotpoints_() {}

  template <int n = Eigen::Dynamic, int m = Eigen::Dynamic>
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
        knotpoints_.push_back(
            std::make_unique<ilqr::KnotPointFunctions<n, m>>(model, costfun));
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

  int NumSegments() const { return N_; }

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
  int N_;
  iLQROptions opts_;
  std::vector<std::unique_ptr<KnotPointFunctionsBase>> knotpoints_;
};

}  // namespace ilqr
}  // namespace altro
