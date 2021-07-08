#pragma once

#include <iostream>

#include "altro/eigentypes.hpp"
#include "altro/utils/derivative_checker.hpp"

namespace altro {
namespace problem {
class CostFunction {
 public:
  virtual ~CostFunction(){};

  /**
   * @brief Evaluate the cost at a single knot point
   *
   * @param x state vector
   * @param u control vector
   * @return double the cost
   */
  virtual double Evaluate(const VectorXdRef& x, const VectorXdRef& u) const = 0;

  /**
   * @brief Evaluate the gradient of the cost at a single knot point
   *
   * @pre the gradient terms @param dx and @param du must be initialized
   * (typically with zeros).
   *
   * @param[in] x (n,) state vector
   * @param[in] u (m,) control vector
   * @param[out] dx (n,) gradient of the cost wrt the state
   * @param[out] du (m,) gradient of the cost wrt the control vector
   */
  virtual void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                        Eigen::Ref<VectorXd> du) const = 0;

  /**
   * @brief Evaluate the Hessian of the cost at a single knot point
   *
   * @pre the Hessian terms @param dxdx, @param dxdu and @param dudu must be
   * initialized
   * (typically with zeros).
   *
   * @param[in] x (n,) state vector
   * @param[in] u (m,) control vector
   * @param[out] dxdx (n,n) Hessian wrt the states
   * @param[out] dxdu (n,m) Hessian cross-term between states and controls
   * @param[out] dudu (m,m) Hessian wrt to the controls
   */
  virtual void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                       Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) const = 0;

  /**
   * @brief Check if the gradient is correct
   *
   * Compares the implemented gradient method to the result obtained using
   * finite differences. The 2-norm error must be less than @param eps.
   *
   * @param x state vector
   * @param u control vector
   * @param eps error tolerance
   * @param eps_fd finite difference step size
   * @return true if gradient is correct
   */
  bool CheckGradient(const VectorXdRef& x, const VectorXdRef& u, const double eps = 1e-4,
                     const double eps_fd = 1e-6, const bool verbose = false) {
    int n = x.size();
    int m = u.size();
    VectorXd z(n + m);
    z << x, u;

    VectorXd dx = VectorXd::Zero(n);
    VectorXd du = VectorXd::Zero(m);
    Gradient(x, u, dx, du);

    auto fz = [&](auto z) -> double { return this->Evaluate(z.head(n), z.tail(m)); };
    VectorXd fd_grad = utils::FiniteDiffGradient(fz, z, eps * eps_fd);

    double err_x = (dx - fd_grad.head(n)).norm();
    double err_u = (du - fd_grad.tail(m)).norm();

    bool pass = (err_x < eps) && (err_u < eps);
    if (verbose) {
      if (pass) {
        VectorXd grad(n + m);
        grad << dx, du;
        std::cout << "Provided:\n" << grad << std::endl;
        std::cout << "Finite Difference:\n" << fd_grad << std::endl;
      }
      std::cout << "Errors (dx, du): " << err_x << ", " << err_u << std::endl;
    }

    return pass;
  }

  /**
   * @brief Check if the Hessian is correct
   *
   * Compares the implemented Hessian method to the result obtained using
   * finite differences. The 2-norm error must be less than @param eps.
   *
   * @param x state vector
   * @param u control vector
   * @param eps error tolerance
   * @param eps_fd finite difference step size
   * @return true if Hessian is correct
   */
  bool CheckHessian(const VectorXdRef& x, const VectorXdRef& u, const double eps = 1e-4,
                    const double eps_fd = 1e-4, const bool verbose = false) {
    int n = x.size();
    int m = u.size();
    VectorXd z(n + m);
    z << x, u;

    MatrixXd dxdx = MatrixXd::Zero(n, n);
    MatrixXd dxdu = MatrixXd::Zero(n, m);
    MatrixXd dudu = MatrixXd::Zero(m, m);
    Hessian(x, u, dxdx, dxdu, dudu);

    auto fz = [&](auto z) -> double { return this->Evaluate(z.head(n), z.tail(m)); };
    MatrixXd fd_hess = utils::FiniteDiffHessian(fz, z, eps_fd);

    double err_xx = (dxdx - fd_hess.topLeftCorner(n, n)).norm();
    double err_xu = (dxdu - fd_hess.topRightCorner(n, m)).norm();
    double err_uu = (dudu - fd_hess.bottomRightCorner(m, m)).norm();

    bool pass = (err_xx < eps) && (err_xu < eps) && (err_uu < eps);
    if (verbose) {
      if (pass) {
        MatrixXd hess(n + m, n + m);
        hess << dxdx, dxdu, dxdu.transpose(), dudu;
        std::cout << "Provided:\n" << hess << std::endl;
        std::cout << "Finite Difference:\n" << fd_hess << std::endl;
      }
      std::cout << "Errors (xx, xu, uu): " << err_xx << ", " << err_xu << ", " << err_uu
                << std::endl;
    }

    return pass;
  }
};

}  // namespace problem
}  // namespace altro