#include <Eigen/Dense>

namespace altro {

class CostExpansion final
{
  using MatrixXd = Eigen::MatrixXd;
 public:
  CostExpansion(int n, int m) : n_(n), m_(m), expansion_(MatrixXd::Zero(n+m, n+m)) {}

  Eigen::MatrixXd::BlockXpr dxdx() { return expansion_.topLeftCorner(n_, n_); }
  Eigen::MatrixXd::BlockXpr dudu() { return expansion_.bottomRightCorner(m_, m_); }
  Eigen::MatrixXd::BlockXpr dxdu() { return expansion_.topRightCorner(n_, m_); }
  Eigen::MatrixXd::BlockXpr dudx() { return expansion_.bottomLeftCorner(m_, n_); }
  MatrixXd& GetExpansion() { return expansion_; }

 private:
  int n_;
  int m_;
  MatrixXd expansion_;
};

class CostFunction
{
 public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  virtual double Evaluate(const VectorXd& x, const VectorXd& u) = 0;
  virtual void Gradient(const VectorXd& x,  const VectorXd& u, 
                        VectorXd& dx, VectorXd& du) = 0;
  virtual void Hessian(const VectorXd& x,  const VectorXd& u, 
                       MatrixXd& dxdx, MatrixXd& dxdu, MatrixXd& dudu) = 0;
};

} // namespace altro