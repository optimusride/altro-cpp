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
