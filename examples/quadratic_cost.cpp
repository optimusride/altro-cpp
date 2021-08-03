#include "examples/quadratic_cost.hpp"

namespace altro {
namespace examples {

double QuadraticCost::Evaluate(const VectorXdRef& x,
                               const VectorXdRef& u) {
  return 0.5 * x.dot(Q_ * x) + x.dot(H_ * u) + 0.5 * u.dot(R_ * u) + q_.dot(x) + r_.dot(u) + c_;
}

void QuadraticCost::Gradient(const VectorXdRef& x,
                             const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                             Eigen::Ref<VectorXd> du) {
  dx = Q_ * x + q_ + H_ * u;
  du = R_ * u + r_ + H_.transpose() * x;
}

void QuadraticCost::Hessian(const VectorXdRef& x,
                            const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                            Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(x);
  ALTRO_UNUSED(u);
  dxdx = Q_;
  dudu = R_;
  dxdu = H_;
}

void QuadraticCost::Validate() {
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
  (void) ispossemidef; // surpress erroneous unused variable error
  for (int i = 0; i < n_; ++i) {
    if (D(i) < 0) {
      ispossemidef = false;
      break;
    }
  }
  ALTRO_ASSERT(ispossemidef, "Q must be positive semi-definite");
}


}  // namespace examples
}  // namespace altro