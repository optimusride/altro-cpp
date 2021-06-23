#pragma once

#include "altro/common/knotpoint.hpp"
#include "altro/common/state_control_sized.hpp"
#include "altro/eigentypes.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/utils/assert.hpp"

namespace altro {
namespace ilqr {

/**
 * @brief Stores the first and second-order expansion of the cost
 * The underlying memory can either be stored on the heap or the stack,
 * depending on whether the state and control dimensions are specified at
 * compile time via the type parameters. Use Eigen::Dynamic to allocate on
 * the heap.
 *
 * @tparam n compile-time state dimension
 * @tparam m compile-time control dimension
 */
template <int n, int m>
class CostExpansion : public StateControlSized<n, m> {
 public:
  explicit CostExpansion(int state_dim, int control_dim)
      : StateControlSized<n, m>(state_dim, control_dim),
        dxdx_(Eigen::Matrix<double, n, n>::Zero(state_dim, state_dim)),
        dxdu_(Eigen::Matrix<double, n, m>::Zero(state_dim, control_dim)),
        dudu_(Eigen::Matrix<double, m, m>::Zero(control_dim, control_dim)),
        dx_(Eigen::Matrix<double, n, 1>::Zero(state_dim, 1)),
        du_(Eigen::Matrix<double, m, 1>::Zero(control_dim, 1)) {}

  Eigen::Matrix<double, n, n>& dxdx() { return dxdx_; }
  Eigen::Matrix<double, n, m>& dxdu() { return dxdu_; }
  Eigen::Matrix<double, m, m>& dudu() { return dudu_; }
  Eigen::Matrix<double, n, 1>& dx() { return dx_; }
  Eigen::Matrix<double, m, 1>& du() { return du_; }

  /**
   * @brief Compute the gradient and Hessian of a cost function and store in
   * the current expansion
   *
   * @tparam n2 compile-time size of the state vector
   * @tparam m2 compile-time size of the control vector
   * @param[in] costfun Cost function whose expansion is to be computed
   * @param[in] z state and control at which to evaluate the expansion
   */
  template <int n2, int m2>
  void CalcExpansion(const problem::CostFunction& costfun,
                     const KnotPoint<n2, m2>& z) {
    CalcExpansion(costfun, z.State(), z.Control());
  }

  void CalcExpansion(const problem::CostFunction& costfun,
                     const Eigen::Ref<const VectorXd>& x,
                     const Eigen::Ref<const VectorXd>& u) {
    ALTRO_ASSERT(x.rows() == this->n_, "Inconsistent state dimension.");
    ALTRO_ASSERT(u.rows() == this->m_, "Inconsistent control dimension.");
    costfun.Gradient(x, u, dx_, du_);
    costfun.Hessian(x, u, dxdx_, dxdu_, dudu_);
  }

 private:
  Eigen::Matrix<double, n, n> dxdx_;
  Eigen::Matrix<double, n, m> dxdu_;
  Eigen::Matrix<double, m, m> dudu_;
  Eigen::Matrix<double, n, 1> dx_;
  Eigen::Matrix<double, m, 1> du_;
};

}  // namespace ilqr
}  // namespace altro
