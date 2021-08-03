#pragma once

#include "altro/eigentypes.hpp"
#include "altro/problem/costfunction.hpp"
#include "altro/utils/utils.hpp"

namespace altro {
namespace examples {

class QuadraticCost : public problem::CostFunction {
 public:
  QuadraticCost(const MatrixXd& Q, const MatrixXd& R, const MatrixXd& H, const VectorXd& q,
                const VectorXd& r, double c = 0, bool terminal = false)
      : n_(q.size()),
        m_(r.size()),
        isblockdiag_(H.norm() < 1e-8),
        Q_(Q),
        R_(R),
        H_(H),
        q_(q),
        r_(r),
        c_(c),
        terminal_(terminal) {
    Validate();
  }

  static QuadraticCost LQRCost(const MatrixXd& Q, const MatrixXd& R, const VectorXd& xref,
                               const VectorXd& uref, bool terminal = false) {
    int n = Q.rows();
    int m = R.rows();
    ALTRO_ASSERT(xref.size() == n, "xref is the wrong size.");
    MatrixXd H = MatrixXd::Zero(n, m);
    VectorXd q = -(Q * xref);
    VectorXd r = -(R * uref);
    double c = 0.5 * xref.dot(Q * xref) + 0.5 * uref.dot(R * uref);
    return QuadraticCost(Q, R, H, q, r, c, terminal);
  }

  int StateDimension() const override { return n_; }
  int ControlDimension() const override { return m_; }
  double Evaluate(const VectorXdRef& x,
                  const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) override;

  const MatrixXd& GetQ() const { return Q_; }
  const MatrixXd& GetR() const { return R_; }
  const MatrixXd& GetH() const { return H_; }
  const VectorXd& Getq() const { return q_; }
  const VectorXd& Getr() const { return r_; }
  double GetConstant() const { return c_; }
  const Eigen::LDLT<MatrixXd>& GetQfact() const { return Qfact_; }
  const Eigen::LLT<MatrixXd>& GetRfact() const { return Rfact_; }
  bool IsBlockDiagonal() const { return isblockdiag_; }

 private:
  void Validate();

  int n_;
  int m_;
  bool isblockdiag_;
  MatrixXd Q_;
  MatrixXd R_;
  MatrixXd H_;
  VectorXd q_;
  VectorXd r_;
  double c_;
  bool terminal_;

  // decompositions of Q and R
  Eigen::LDLT<MatrixXd> Qfact_;
  Eigen::LLT<MatrixXd> Rfact_;
};

}  // namespace examples
}  // namespace altro