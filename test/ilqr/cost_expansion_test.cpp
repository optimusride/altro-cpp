#include <gtest/gtest.h>

#include "altro/ilqr/cost_expansion.hpp"
#include "examples/quadratic_cost.hpp"

namespace altro {
namespace ilqr {
class CostExpansionTest : public ::testing::Test {
 public:
  int n = 4;
  int m = 3;
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
  auto bad_state = [&]() { 
  	CostExpansion<4, 3> expansion(n-1, m);
  };
  EXPECT_DEATH(bad_state(), "Assert.*State sizes must be consistent");

  auto bad_control= [&]() { 
  	CostExpansion<4, 3> expansion(n, m+1);
  };
  EXPECT_DEATH(bad_control(), "Assert.*Control sizes must be consistent");
}


TEST_F(CostExpansionTest, GetParts) {
  CostExpansion<4, 3> expansion(n, m);
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
  CostExpansion<-1, -1> expansion(n, m);
	examples::QuadraticCost costfun(Q, R, H, q, r, c);
	KnotPoint<-1,-1> z = KnotPoint<-1,-1>::Random(n, m);
	KnotPoint<4,3> z_static = KnotPoint<4,3>::Random(n, m);

	// try with a dynamically-sized knot point
	expansion.CalcExpansion(costfun, z);
	EXPECT_TRUE(expansion.dx().isApprox(Q*z.State() + q));
	EXPECT_TRUE(expansion.du().isApprox(R*z.Control() + r));
	EXPECT_TRUE(expansion.dxdx().isApprox(Q));
	EXPECT_TRUE(expansion.dudu().isApprox(R));
	EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));

	// Try with a statically-sized knot point
	expansion.CalcExpansion(costfun, z_static);
	EXPECT_TRUE(expansion.dx().isApprox(Q*z_static.State() + q));
	EXPECT_TRUE(expansion.du().isApprox(R*z_static.Control() + r));
	EXPECT_TRUE(expansion.dxdx().isApprox(Q));
	EXPECT_TRUE(expansion.dudu().isApprox(R));
	EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));
}

TEST_F(CostExpansionTest, QuadraticCostExpansionStatic) {
	CostExpansion<4,3> expansion(n, m);
	examples::QuadraticCost costfun(Q, R, H, q, r, c);
	KnotPoint<-1,-1> z = KnotPoint<-1,-1>::Random(n, m);
	KnotPoint<4,3> z_static = KnotPoint<4,3>::Random(n, m);

	// try with a dynamically-sized knot point
	expansion.CalcExpansion(costfun, z);
	EXPECT_TRUE(expansion.dx().isApprox(Q*z.State() + q));
	EXPECT_TRUE(expansion.du().isApprox(R*z.Control() + r));
	EXPECT_TRUE(expansion.dxdx().isApprox(Q));
	EXPECT_TRUE(expansion.dudu().isApprox(R));
	EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));

	// Try with a statically-sized knot point
	expansion.CalcExpansion(costfun, z_static);
	EXPECT_TRUE(expansion.dx().isApprox(Q*z_static.State() + q));
	EXPECT_TRUE(expansion.du().isApprox(R*z_static.Control() + r));
	EXPECT_TRUE(expansion.dxdx().isApprox(Q));
	EXPECT_TRUE(expansion.dudu().isApprox(R));
	EXPECT_TRUE(expansion.dxdu().isApproxToConstant(0));

	EXPECT_EQ(expansion.StateDimension(), n);
	EXPECT_EQ(expansion.ControlDimension(), m);
	EXPECT_EQ(expansion.StateMemorySize(), n);
	EXPECT_EQ(expansion.ControlMemorySize(), m);
	constexpr int n_mem = CostExpansion<4,3>::StateMemorySize();
	Vector<n_mem> x_test;
	EXPECT_EQ(n_mem, n);
	EXPECT_EQ(x_test.SizeAtCompileTime, n);
}

}  // namespace ilqr
}  // namespace altro