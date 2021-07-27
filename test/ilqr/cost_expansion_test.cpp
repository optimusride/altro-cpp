#include <gtest/gtest.h>

#include "altro/ilqr/cost_expansion.hpp"
#include "altro/utils/assert.hpp"
#include "examples/quadratic_cost.hpp"

namespace altro {
namespace ilqr {

constexpr int n_static = 4;
constexpr int m_static = 3;
constexpr int HEAP = Eigen::Dynamic;

class CostExpansionTest : public ::testing::Test {
 public:
  int n = n_static;
  int m = m_static;
  MatrixXd Q = Eigen::Vector4d(1, 2, 3, 4).asDiagonal();
  MatrixXd R = Eigen::Vector3d(5, 6, 7).asDiagonal();
  MatrixXd H = Eigen::MatrixXd::Zero(4, 3);
  VectorXd q = Eigen::Vector4d::LinSpaced(4, 8, 11);
  VectorXd r = Eigen::Vector3d::LinSpaced(3, 12, 14);
  double c = 1.0;

 protected:
  void SetUp() override {}
};

TEST_F(CostExpansionTest, Construction) {
  CostExpansion<4, 3> expansion(n, m);
  CostExpansion<Eigen::Dynamic, Eigen::Dynamic> expansion2(n, m);
  EXPECT_EQ(expansion.StateDimension(), n);
  EXPECT_EQ(expansion2.StateDimension(), n);
  EXPECT_EQ(expansion.ControlDimension(), m);
  EXPECT_EQ(expansion2.ControlDimension(), m);
}

TEST_F(CostExpansionTest, ConstructionDeath) {
  if (utils::AssertionsActive()) {
    auto bad_state = [&]() { CostExpansion<4, 3> expansion(n - 1, m); };
    EXPECT_DEATH(bad_state(), "Assert.*State sizes must be consistent");
  
    auto bad_control = [&]() { CostExpansion<4, 3> expansion(n, m + 1); };
    EXPECT_DEATH(bad_control(), "Assert.*Control sizes must be consistent");
  }
}

TEST_F(CostExpansionTest, Copy) {
  Q = Q.transpose() * Q;
  MatrixXd Qxx = Q.topLeftCorner(n, n);
  MatrixXd Qxu = Q.topRightCorner(n, m);
  MatrixXd Qux = Q.bottomLeftCorner(m, n);
  MatrixXd Quu = Q.bottomRightCorner(m, m);
  VectorXd Qx = VectorXd::Random(n);
  VectorXd Qu = VectorXd::Random(m);

  CostExpansion<n_static, m_static> expansion(n, m);
  expansion.dxdx() = Qxx;
  expansion.dxdu() = Qxu;
  expansion.dudu() = Quu;
  expansion.dx() = Qx;
  expansion.du() = Qu;

  CostExpansion<n_static, m_static> expansion_copy(expansion);
  EXPECT_TRUE(expansion_copy.dxdx().isApprox(Qxx));
  EXPECT_TRUE(expansion_copy.dxdu().isApprox(Qxu));
  EXPECT_TRUE(expansion_copy.dudu().isApprox(Quu));
  EXPECT_TRUE(expansion_copy.dx().isApprox(Qx));
  EXPECT_TRUE(expansion_copy.du().isApprox(Qu));

  expansion.dxdx() *= 2;
  expansion.dxdu() *= 3;
  expansion.du() *= 4;

  // Make sure it doesn't modify the copy
  EXPECT_TRUE(expansion_copy.dxdx().isApprox(Qxx));
  EXPECT_TRUE(expansion_copy.dxdu().isApprox(Qxu));
  EXPECT_TRUE(expansion_copy.du().isApprox(Qu));
  expansion_copy = expansion;
  EXPECT_TRUE(expansion_copy.dxdx().isApprox(2 * Qxx));
  EXPECT_TRUE(expansion_copy.dxdu().isApprox(3 * Qxu));
  EXPECT_TRUE(expansion_copy.du().isApprox(4 * Qu));
}

TEST_F(CostExpansionTest, GetParts) {
  CostExpansion<n_static, m_static> expansion(n, m);
  VectorXd du = expansion.du();
  EXPECT_EQ(du.rows(), m);
  expansion.dxdx().topLeftCorner(2, 2) = Eigen::Matrix2d::Constant(10);
  MatrixXd dxdx_expected(n, n);
  // clang-format off
	dxdx_expected << 10,10,0,0,
	                 10,10,0,0,
									 0, 0, 0,0,
									 0, 0, 0,0;
  // clang-format on
  EXPECT_TRUE(expansion.dxdx().isApprox(dxdx_expected));

  EXPECT_EQ(expansion.dxdu().rows(), n);
  EXPECT_EQ(expansion.dxdu().cols(), m);
}

TEST_F(CostExpansionTest, QuadraticCostExpansionDynamic) {
  CostExpansion<HEAP, HEAP> expansion(n, m);
  examples::QuadraticCost costfun(Q, R, H, q, r, c);
  KnotPoint<HEAP, HEAP> z = KnotPoint<HEAP, HEAP>::Random(n, m);
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random(n, m);

  // try with a dynamically-sized knot point
  expansion.CalcExpansion(costfun, z);
  EXPECT_TRUE(expansion.dx().isApprox(Q * z.State() + q));
  EXPECT_TRUE(expansion.du().isApprox(R * z.Control() + r));
  EXPECT_TRUE(expansion.dxdx().isApprox(Q));
  EXPECT_TRUE(expansion.dudu().isApprox(R));
  EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));

  // Try with a statically-sized knot point
  expansion.CalcExpansion(costfun, z_static);
  EXPECT_TRUE(expansion.dx().isApprox(Q * z_static.State() + q));
  EXPECT_TRUE(expansion.du().isApprox(R * z_static.Control() + r));
  EXPECT_TRUE(expansion.dxdx().isApprox(Q));
  EXPECT_TRUE(expansion.dudu().isApprox(R));
  EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));
}

TEST_F(CostExpansionTest, QuadraticCostExpansionStatic) {
  CostExpansion<n_static, m_static> expansion(n, m);
  examples::QuadraticCost costfun(Q, R, H, q, r, c);
  KnotPoint<HEAP, HEAP> z = KnotPoint<HEAP, HEAP>::Random(n, m);
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random(n, m);

  // try with a dynamically-sized knot point
  expansion.CalcExpansion(costfun, z);
  EXPECT_TRUE(expansion.dx().isApprox(Q * z.State() + q));
  EXPECT_TRUE(expansion.du().isApprox(R * z.Control() + r));
  EXPECT_TRUE(expansion.dxdx().isApprox(Q));
  EXPECT_TRUE(expansion.dudu().isApprox(R));
  EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));

  // Try with a statically-sized knot point
  expansion.CalcExpansion(costfun, z_static);
  EXPECT_TRUE(expansion.dx().isApprox(Q * z_static.State() + q));
  EXPECT_TRUE(expansion.du().isApprox(R * z_static.Control() + r));
  EXPECT_TRUE(expansion.dxdx().isApprox(Q));
  EXPECT_TRUE(expansion.dudu().isApprox(R));
  EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));

  EXPECT_EQ(expansion.StateDimension(), n);
  EXPECT_EQ(expansion.ControlDimension(), m);
  EXPECT_EQ(expansion.StateMemorySize(), n);
  EXPECT_EQ(expansion.ControlMemorySize(), m);
  constexpr int n_mem = CostExpansion<n_static, m_static>::StateMemorySize();
  VectorN<n_mem> x_test;
  EXPECT_EQ(n_mem, n);
  EXPECT_EQ(x_test.SizeAtCompileTime, n);
}

}  // namespace ilqr
}  // namespace altro