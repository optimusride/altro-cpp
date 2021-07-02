#pragma once

#include "altro/eigentypes.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace problem {

class Dynamics {
public:
  virtual ~Dynamics(){};

  virtual int StateDimension() const = 0;
  virtual int ControlDimension() const = 0;

  /**
   * @brief Indicate whether Hessian is defined.
   *
   * @return true Hessian is defined
   * @return false Hessian is not defined
   */
  virtual bool HasHessian() const = 0;
};

class ContinuousDynamics : public Dynamics {
public:
  virtual ~ContinuousDynamics(){};

  /**
   * @brief Evaluate the continuous-time dynamics
   *
   * @param x state vector
   * @param u control vector
   * @param t independent variable (e.g. time)
   * @return VectorXd the state derivative
   */
  virtual VectorXd Evaluate(const VectorXd &x, const VectorXd &u,
                            const float t) const {
    VectorXd xdot(x.rows());
    EvaluateInplace(x, u, t, xdot);
    return xdot;
  }
  VectorXd operator()(const VectorXd &x, const VectorXd &u,
                      const float t) const {
    return Evaluate(x, u, t);
  }

  virtual void EvaluateInplace(const Eigen::Ref<const VectorXd> &x,
                               const Eigen::Ref<const VectorXd> &u,
                               const float t,
                               Eigen::Ref<VectorXd> xdot) const = 0;

  /**
   * @brief Evaluate the nxm continuous dynamics Jacobian
   *
   * User must supply a pre-initialized Jacobian matrix.
   *
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[out] jac dense (n,m) dynamics Jacobian
  */
  virtual void Jacobian(const Eigen::Ref<const VectorXd> &x,
                        const Eigen::Ref<const VectorXd> &u, const float t,
                        Eigen::Ref<MatrixXd> jac) const = 0;

  /**
   * @brief Evaluate the derivative of the Jacobian-transpose vector product:
   * d/dx(J^T b).
   *
   * Not made pure virtual since it doesn't have to be defined.
   *
   * User must supply a pre-initialized Hessian matrix.
   *
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] b vector multiplying the Jacobian transpose (dimension n)
   * @param[out] hess nxm derivative of the Jacobian-tranpose vector product
   */
  virtual void Hessian(const Eigen::Ref<const VectorXd> &x,
                       const Eigen::Ref<const VectorXd> &u, const float t,
                       const Eigen::Ref<const VectorXd> &b,
                       Eigen::Ref<MatrixXd> hess) const {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  bool CheckJacobian(const double eps = 1e-4) const {
    const int n = StateDimension();
    const int m = ControlDimension();
    VectorXd x = VectorXd::Random(n);
    VectorXd u = VectorXd::Random(m);
    float t = static_cast<float>(rand()) / RAND_MAX;
    return CheckJacobian(x, u, t, eps);
  }

  bool CheckJacobian(const Eigen::Ref<const VectorXd> &x,
                     const Eigen::Ref<const VectorXd> &u, const float t,
                     const double eps = 1e-4) const {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd z(n + m);
    z << x, u;

    // Calculate Jacobian
    MatrixXd jac = MatrixXd::Zero(n, n + m);
    Jacobian(x, u, t, jac);

    // Calculate using finite differencing
    auto fz = [&](auto z) -> VectorXd {
      return this->Evaluate(z.head(n), z.tail(m), t);
    };
    auto fd_jac =
        utils::FiniteDiffJacobian<Eigen::Dynamic, Eigen::Dynamic>(fz, z);

    // Compare
    double err = (fd_jac - jac).norm();
    return err < eps;
  }

  bool CheckHessian(const double eps = 1e-4) {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd x = VectorXd::Random(n);
    VectorXd u = VectorXd::Random(m);
    VectorXd b = VectorXd::Random(n);
    float t = static_cast<float>(rand()) / RAND_MAX;
    return CheckHessian(x, u, t, b, eps);
  }
  bool CheckHessian(const Eigen::Ref<const VectorXd> &x,
                    const Eigen::Ref<const VectorXd> &u, const float t,
                    const Eigen::Ref<const VectorXd> &b,
                    const double eps = 1e-4) {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd z(n + m);
    z << x, u;

    MatrixXd hess(n + m, n + m);
    Hessian(x, u, t, b, hess);

    auto jvp = [&](auto z) -> double {
      return this->Evaluate(z.head(n), z.tail(m), t).transpose() * b;
    };
    MatrixXd fd_hess = utils::FiniteDiffHessian(jvp, z);

    double err = (fd_hess - hess).norm();
    return err < eps;
  }
};

class DiscreteDynamics : public Dynamics {
public:
  virtual ~DiscreteDynamics(){};

  /**
   * @brief Evaluate the discrete-time dynamics
   *
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h segment length (e.g. time step)
   * @return VectorXd the next state vector
   */
  virtual VectorXd Evaluate(const VectorXd &x, const VectorXd &u, const float t,
                            const float h) const {
    VectorXd xnext(x.rows());
    EvaluateInplace(x, u, t, h, xnext);
    return xnext;
  }
  VectorXd operator()(const VectorXd &x, const VectorXd &u, const float t,
                      const float h) const {
    return Evaluate(x, u, t, h);
  }

  virtual void EvaluateInplace(const Eigen::Ref<const VectorXd> &x,
                               const Eigen::Ref<const VectorXd> &u,
                               const float t, const float h,
                               Eigen::Ref<VectorXd> xnext) const = 0;

  /**
   * @brief Evaluate the nxm discrete dynamics Jacobian
   *
   * User must supply a pre-initialized Jacobian matrix.
   *
   * @param[in] x (n,) state vector (dimension n)
   * @param[in] u (m,) control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] h segment length (e.g. time step)
   * @param[out] jac dense (n,m) dynamics Jacobian
   */
  virtual void Jacobian(const Eigen::Ref<const VectorXd> &x,
                        const Eigen::Ref<const VectorXd> &u, const float t,
                        const float h, Eigen::Ref<MatrixXd> jac) const = 0;

  /**
   * @brief Evaluate the derivative of the Jacobian-transpose vector product:
   * d/dx(J^T b).
   *
   * Not made pure virtual since it doesn't have to be defined.
   *
   * User must supply a pre-initialized Hessian matrix.
   *
   * @param[in] x (n,) state vector (dimension n)
   * @param[in] u (m,) control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] b (n,) vector multiplying the Jacobian transpose (dimension n)
   * @param hvp nxm derivative of the Jacobian-tranpose vector product
   */
  virtual void Hessian(const Eigen::Ref<const VectorXd> &x,
                       const Eigen::Ref<const VectorXd> &u, const float t,
                       const float h, const Eigen::Ref<const VectorXd> &b,
                       Eigen::Ref<MatrixXd> hess) const {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(h);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  bool CheckJacobian(const double eps = 1e-4) const {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd x = VectorXd::Random(n);
    VectorXd u = VectorXd::Random(m);
    float t = static_cast<float>(rand()) / RAND_MAX;
    float h = 0.1;
    return CheckJacobian(x, u, t, h, eps);
  }

  bool CheckJacobian(const Eigen::Ref<const VectorXd> &x,
                     const Eigen::Ref<const VectorXd> &u, const float t,
                     const float h, const double eps = 1e-4) const {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd z(n + m);
    z << x, u;

    // Calculate Jacobian
    MatrixXd jac = MatrixXd::Zero(n, n + m);
    Jacobian(x, u, t, h, jac);

    // Calculate using finite differencing
    auto fz = [&](auto z) -> VectorXd {
      return this->Evaluate(z.head(n), z.tail(m), t, h);
    };
    auto fd_jac =
        utils::FiniteDiffJacobian<Eigen::Dynamic, Eigen::Dynamic>(fz, z);

    // Compare
    double err = (fd_jac - jac).norm();
    return err < eps;
  }

  bool CheckHessian(const double eps = 1e-4) {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd x = VectorXd::Random(n);
    VectorXd u = VectorXd::Random(m);
    VectorXd b = VectorXd::Random(n);
    float t = static_cast<float>(rand()) / RAND_MAX;
    float h = 0.1;
    return CheckHessian(x, u, t, h, b, eps);
  }

  bool CheckHessian(const Eigen::Ref<const VectorXd> &x,
                    const Eigen::Ref<const VectorXd> &u, const float t,
                    const float h, const Eigen::Ref<const VectorXd> &b,
                    const double eps = 1e-4) {
    int n = StateDimension();
    int m = ControlDimension();
    VectorXd z(n + m);
    z << x, u;

    MatrixXd hess(n + m, n + m);
    Hessian(x, u, t, h, b, hess);

    auto jvp = [&](auto z) -> double {
      return this->Evaluate(z.head(n), z.tail(m), t, h).transpose() * b;
    };
    MatrixXd fd_hess = utils::FiniteDiffHessian(jvp, z);

    double err = (fd_hess - hess).norm();
    return err < eps;
  }
};

} // namespace problem
} // namespace altro
