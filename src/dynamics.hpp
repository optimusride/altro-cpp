#pragma once

#include "eigentypes.hpp"
namespace altro {

class ContinuousDynamics 
{
 public:

  virtual ~ContinuousDynamics() {};

  /**
   * @brief Evaluate the continuous-time dynamics
   * 
   * @param x state vector
   * @param u control vector
   * @param t independent variable (e.g. time)
   * @return VectorXd the state derivative
   */
  virtual VectorXd Evaluate(const VectorXd& x, const VectorXd& u, const float t) = 0;

  /**
   * @brief Evaluate the nxm continuous dynamics Jacobian
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[out] jac nxm dense dynamics Jacobian
   */
  virtual void Jacobian(const VectorXd& x, const VectorXd& u, const float t, MatrixXd jac) = 0;

  /**
   * @brief Evaluate the derivative of the Jacobian-transpose vector product: d/dx(J^T b).
   * 
   * Not made pure virtual since it doesn't have to be defined.
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] b vector multiplying the Jacobian transpose (dimension n)
   * @param hvp nxm derivative of the Jacobian-tranpose vector product
   */
  virtual void Hessian(const VectorXd& x, const VectorXd& u, const float t, 
                                       const VectorXd& b, MatrixXd hess);

  /**
   * @brief Indicate whether Hessian is defined.
   * 
   * @return true Hessian is defined 
   * @return false Hessian is not defined
   */
  virtual bool HasHessian() = 0;
};

class DiscreteDynamics
{
 public:

  virtual ~DiscreteDynamics() {};

  /**
   * @brief Evaluate the discrete-time dynamics 
   * 
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h segment length (e.g. time step)
   * @return VectorXd the next state vector
   */
  virtual VectorXd Evaluate(const VectorXd& x, const VectorXd& u, const float t, const float h) = 0;

  /**
   * @brief Evaluate the nxm discrete dynamics Jacobian
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] h segment length (e.g. time step)
   * @param[out] jac nxm dense dynamics Jacobian
   */
  virtual void Jacobian(const VectorXd& x, const VectorXd& u, const float t, const float h,
                        MatrixXd jac) = 0;

  /**
   * @brief Evaluate the derivative of the Jacobian-transpose vector product: d/dx(J^T b).
   * 
   * Not made pure virtual since it doesn't have to be defined.
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] b vector multiplying the Jacobian transpose (dimension n)
   * @param hvp nxm derivative of the Jacobian-tranpose vector product
   */
  virtual void Hessian(const VectorXd& x, const VectorXd& u, const float t, const float h,
                                       const VectorXd& b, MatrixXd hess);

  /**
   * @brief Indicate whether Hessian is defined.
   * 
   * @return true Hessian is defined 
   * @return false Hessian is not defined
   */
  virtual bool HasHessian() = 0;
  virtual size_t StateDimension() = 0;
  virtual size_t ControlDimension() = 0;
};

} // namespace altro