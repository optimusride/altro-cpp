#pragma once

#include "eigentypes.hpp"
namespace altro {

class CostExpansion final
{
 public:
  CostExpansion(int n, int m) : n_(n), m_(m), expansion_(MatrixXd::Zero(n+m, n+m)) {}

  MatrixXd::BlockXpr dxdx() { return expansion_.topLeftCorner(n_, n_); }
  MatrixXd::BlockXpr dudu() { return expansion_.bottomRightCorner(m_, m_); }
  MatrixXd::BlockXpr dxdu() { return expansion_.topRightCorner(n_, m_); }
  MatrixXd::BlockXpr dudx() { return expansion_.bottomLeftCorner(m_, n_); }
  MatrixXd& GetExpansion() { return expansion_; }

 private:
  int n_;
  int m_;
  MatrixXd expansion_;
};

class CostFunction
{
 public:
  virtual ~CostFunction() {};

  /**
   * @brief Evaluate the cost as a single knot point
   * 
   * @param x 
   * @param u 
   * @return double 
   */
  virtual double Evaluate(const VectorXd& x, const VectorXd& u) const = 0;
  virtual void Gradient(const VectorXd& x, const VectorXd& u, 
                        VectorXd& dx, VectorXd& du) const = 0;
  virtual void Hessian(const VectorXd& x, const VectorXd& u, 
                       MatrixXd& dxdx, MatrixXd& dxdu, MatrixXd& dudu) const = 0;

};

} // namespace altro