#pragma once

#include "eigentypes.hpp"
#include "utils/utils.hpp"
namespace altro {

class ContinuousDynamics 
{
 public:

  virtual ~ContinuousDynamics() {}; 

  virtual int StateDimension() const = 0;
  virtual int ControlDimension() const = 0;

  /**
   * @brief Evaluate the continuous-time dynamics
   * 
   * @param x state vector
   * @param u control vector
   * @param t independent variable (e.g. time)
   * @return VectorXd the state derivative
   */
  virtual VectorXd Evaluate(const VectorXd& x, const VectorXd& u, const float t) const = 0;
  VectorXd operator()(const VectorXd& x, const VectorXd& u, const float t) const
  { 
    return Evaluate(x, u, t); 
  }

  /**
   * @brief Evaluate the nxm continuous dynamics Jacobian
   * 
   * User must supply a pre-initialized Jacobian matrix.
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[out] jac nxm dense dynamics Jacobian
   */
  virtual void Jacobian(const VectorXd& x, const VectorXd& u, 
                        const float t, MatrixXd& jac) const = 0;

  /**
   * @brief Evaluate the derivative of the Jacobian-transpose vector product: d/dx(J^T b).
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
  virtual void Hessian(const VectorXd& x, const VectorXd& u, const float t, 
                       const VectorXd& b, MatrixXd& hess) const
  {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  /**
   * @brief Indicate whether Hessian is defined.
   * 
   * @return true Hessian is defined 
   * @return false Hessian is not defined
   */
  virtual bool HasHessian() const = 0;

};

class DiscreteDynamics
{
 public:

  virtual ~DiscreteDynamics() {};

  virtual int StateDimension() const = 0;
  virtual int ControlDimension() const = 0;


  /**
   * @brief Evaluate the discrete-time dynamics 
   * 
   * @param[in] x state vector
   * @param[in] u control vector
   * @param[in] t independent variable (e.g. time)
   * @param[in] h segment length (e.g. time step)
   * @return VectorXd the next state vector
   */
  virtual VectorXd Evaluate(const VectorXd& x, const VectorXd& u, 
                            const float t, const float h) const = 0;
  VectorXd operator()(const VectorXd& x, const VectorXd& u, 
                      const float t, const float h) const { 
    return Evaluate(x, u, t, h); 
  }

  /**
   * @brief Evaluate the nxm discrete dynamics Jacobian
   * 
   * User must supply a pre-initialized Jacobian matrix.
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] h segment length (e.g. time step)
   * @param[out] jac nxm dense dynamics Jacobian
   */
  virtual void Jacobian(const VectorXd& x, const VectorXd& u, 
                        const float t, const float h, MatrixXd& jac) const = 0;

  /**
   * @brief Evaluate the derivative of the Jacobian-transpose vector product: d/dx(J^T b).
   * 
   * Not made pure virtual since it doesn't have to be defined.
   * 
   * User must supply a pre-initialized Hessian matrix.
   * 
   * @param[in] x state vector (dimension n)
   * @param[in] u control vector (dimension m)
   * @param[in] t independent variable (e.g. time)
   * @param[in] b vector multiplying the Jacobian transpose (dimension n)
   * @param hvp nxm derivative of the Jacobian-tranpose vector product
   */
  virtual void Hessian(const VectorXd& x, const VectorXd& u, 
                       const float t, const float h,
                       const VectorXd& b, MatrixXd& hess) const {
    ALTRO_UNUSED(x);
    ALTRO_UNUSED(u);
    ALTRO_UNUSED(t);
    ALTRO_UNUSED(h);
    ALTRO_UNUSED(b);
    ALTRO_UNUSED(hess);
  }

  /**
   * @brief Indicate whether Hessian is defined.
   * 
   * @return true Hessian is defined 
   * @return false Hessian is not defined
   */
  virtual bool HasHessian() const = 0;
};


} // namespace altro