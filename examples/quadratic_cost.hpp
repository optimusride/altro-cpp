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

  double Evaluate(const VectorXdRef& x,
                  const VectorXdRef& u) const override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u,
                Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) const override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u,
               Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dxdu,
               Eigen::Ref<MatrixXd> dudu) const override;

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
  void Validate() {
    ALTRO_ASSERT(Q_.rows() == n_, "Q has the wrong number of rows");
    ALTRO_ASSERT(Q_.cols() == n_, "Q has the wrong number of columns");
    ALTRO_ASSERT(R_.rows() == m_, "R has the wrong number of rows");
    ALTRO_ASSERT(R_.cols() == m_, "R has the wrong number of columns");
    ALTRO_ASSERT(H_.rows() == n_, "H has the wrong number of rows");
    ALTRO_ASSERT(H_.cols() == m_, "H has the wrong number of columns");

    // Check symmetry of Q and R
    ALTRO_ASSERT(Q_.isApprox(Q_.transpose()), "Q is not symmetric");
    ALTRO_ASSERT(R_.isApprox(R_.transpose()), "R is not symmetric");

    // Check that R is positive definite
    if (!terminal_) {
      Rfact_.compute(R_);
      ALTRO_ASSERT(Rfact_.info() == Eigen::Success, "R must be positive definite");
    }

    // Check if Q is positive semidefinite
    Qfact_.compute(Q_);
    ALTRO_ASSERT(Qfact_.info() == Eigen::Success,
                 "The LDLT decomposition could of Q could not be computed. "
                 "Must be positive semi-definite");
    Eigen::Diagonal<const MatrixXd> D = Qfact_.vectorD();
    bool ispossemidef = true;
    for (int i = 0; i < n_; ++i) {
      if (D(i) < 0) {
        ispossemidef = false;
        break;
      }
    }
    ALTRO_ASSERT(ispossemidef, "Q must be positive semi-definite");
  }

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