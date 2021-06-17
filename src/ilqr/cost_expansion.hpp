#include "eigentypes.hpp"
#include "problem/costfunction.hpp"
#include "common/knotpoint.hpp"

namespace altro {
namespace ilqr {

template <int n, int m>
class CostExpansion {
 public:
  explicit CostExpansion(int state_dim, int control_dim)
      : n_(state_dim),
        m_(control_dim),
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

  void CalcExpansion(const problem::CostFunction& costfun, const KnotPoint<n,m>& z) {
    Vector<n> x = z.State();
    Vector<m> u = z.State();
    costfun.Gradient(x, u, dx_, du_);
    costfun.Hessian(x, u, dxdx_, dxdu_, dudu_);
  }

 private:
  int n_;
  int m_;
  Eigen::Matrix<double, n, n> dxdx_;
  Eigen::Matrix<double, n, m> dxdu_;
  Eigen::Matrix<double, m, m> dudu_;
  Eigen::Matrix<double, n, 1> dx_;
  Eigen::Matrix<double, m, 1> du_;
};

}	// namespace ilqr
} // namespace altro
