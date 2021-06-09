#include <gtest/gtest.h>
#include "quadraticcost.hpp"
#include "eigentypes.hpp" 
#include <iostream>

namespace altro {
namespace examples {

class QuadraticCostTest : public ::testing::Test 
{
 protected:
  void SetUp() override {
    Q = Eigen::Vector3d(1,2,3).asDiagonal(); 
    R = Eigen::Vector2d(4,5).asDiagonal();
    H = MatrixXd::Zero(3,2);
    q = Eigen::Vector3d(6,7,8);
    r = Eigen::Vector2d(9,10);
    c = 11;
  }
  MatrixXd Q; 
  MatrixXd R; 
  MatrixXd H; 
  VectorXd q; 
  VectorXd r; 
  double c;
    
};

TEST_F(QuadraticCostTest, Construction)
{
  QuadraticCost qcost = QuadraticCost(Q, R, H, q, r, c);
  EXPECT_TRUE(qcost.GetQ().isApprox(Q));
  EXPECT_TRUE(qcost.GetR().isApprox(R));
  EXPECT_TRUE(qcost.GetH().isApprox(H));
  EXPECT_TRUE(qcost.Getq().isApprox(q));
  EXPECT_TRUE(qcost.Getr().isApprox(r));
  EXPECT_DOUBLE_EQ(qcost.GetConstant(), c);

  EXPECT_TRUE(qcost.IsBlockDiagonal());

  H(0,0) = 2;
  QuadraticCost qcost2 = QuadraticCost(Q, R, H, q, r, c);
  EXPECT_FALSE(qcost2.IsBlockDiagonal());
}

TEST_F(QuadraticCostTest, ConstructionPSD)
{
  // Make Q PSD
  Q(0,0) = 0;
  QuadraticCost qcost = QuadraticCost(Q, R, H, q, r, c);
  EXPECT_TRUE(qcost.GetQ().isApprox(Q));
  auto Qfact = qcost.GetQfact();
  EXPECT_DOUBLE_EQ(Qfact.vectorD().minCoeff(), 0);
}

TEST_F(QuadraticCostTest, ConstructionDeath)
{
  // Make R PSD
  R(0,0) = 0;
  EXPECT_DEATH(QuadraticCost(Q, R, H, q, r, c), "Assert.*R.*positive definite");

  // Revert back and check again
  R(0,0) = 1;
  QuadraticCost qcost = QuadraticCost(Q, R, H, q, r, c);

  // Make Q not semi-definite
  Q(0,0) = -1;
  EXPECT_DEATH(QuadraticCost(Q, R, H, q, r, c), "Assert.*Q.*positive semi-definite");

  // Check sizes
  VectorXd q2 = Eigen::Vector2d(10,12);                     // wrong size
  VectorXd r2 = Eigen::Vector3d(10,12,13);                  // wrong size
  MatrixXd Q2 = (MatrixXd(2,2) << 1,2,3,4).finished();      // wrong size
  MatrixXd Q3 = (MatrixXd(2,3) << 1,2,3,4,5,6).finished();  // not square
  EXPECT_DEATH(QuadraticCost(Q, R, H, q2, r, c), "Assert.*wrong number of");
  EXPECT_DEATH(QuadraticCost(Q, R, H, q, r2, c), "Assert.*wrong number of");
  EXPECT_DEATH(QuadraticCost(Q2, R, H, q, r, c), "Assert.*wrong number of");
  EXPECT_DEATH(QuadraticCost(Q3, R, H, q, r, c), "Assert.*wrong number of");

  // Check symmetry
  MatrixXd R2 = (MatrixXd(2,2) << 1,1,0,2).finished();      // asymmetric
  EXPECT_DEATH(QuadraticCost(Q, R2, H, q, r, c), "Assert.*not symmetric");
}

TEST_F(QuadraticCostTest, Evaluation)
{
  QuadraticCost qcost = QuadraticCost(Q, R, H, q, r, c);
  VectorXd x = VectorXd::Constant(3, 1.0);
  VectorXd u = VectorXd::Constant(2, 1.0);
  double J = Q.sum() + R.sum() + q.sum() + r.sum() + c; 
  EXPECT_DOUBLE_EQ(J, qcost.Evaluate(x, u));

  x = VectorXd::Random(3);
  u = VectorXd::Random(2);
  J = x.dot(Q*x) + u.dot(R*u) + q.dot(x) + r.dot(u) + c;
  EXPECT_DOUBLE_EQ(J, qcost.Evaluate(x, u));

  VectorXd dx = VectorXd::Zero(3);
  VectorXd du = VectorXd::Zero(2);
  qcost.Gradient(x, u, dx, du);
  EXPECT_TRUE(dx.isApprox(Q*x + q));
  EXPECT_TRUE(du.isApprox(R*u + r));

  MatrixXd dxdx = MatrixXd::Zero(3,3);
  MatrixXd dxdu = MatrixXd::Zero(3,2);
  MatrixXd dudu = MatrixXd::Zero(2,2);
  qcost.Hessian(x,u, dxdx, dxdu, dudu);
  EXPECT_TRUE(dxdx.isApprox(Q));
  EXPECT_TRUE(dxdu.isApprox(H));
  EXPECT_TRUE(dudu.isApprox(R));
}

TEST_F(QuadraticCostTest, LQRCost)
{
  VectorXd xref = Eigen::Vector3d(-1,-2,-3);
  VectorXd uref = Eigen::Vector2d(-4,-5);
  QuadraticCost qcost = QuadraticCost::LQRCost(Q, R, xref, uref);
  EXPECT_TRUE(qcost.Getq().isApprox(-2*Q*xref));
  EXPECT_TRUE(qcost.Getr().isApprox(-2*R*uref));

  VectorXd x = VectorXd::Random(3);
  VectorXd u = VectorXd::Random(2);
  VectorXd dx = x - xref;
  VectorXd du = u - uref;
  double J = dx.dot(Q*dx) + du.dot(R*du);
  EXPECT_DOUBLE_EQ(J, qcost.Evaluate(x,u));
}

} // namespace examples
} // namespace altro