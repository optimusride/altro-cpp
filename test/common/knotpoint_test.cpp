#include <gtest/gtest.h>

#include "altro/common/knotpoint.hpp"

namespace altro {

TEST(KnotPointTest, Constructor) {
  KnotPoint<3,2> z;
	EXPECT_TRUE(z.State().isApprox(Vector<3>::Zero()));
	EXPECT_TRUE(z.Control().isApprox(Vector<2>::Zero()));

	Vector<3> x = Vector<3>::Constant(1);
	Vector<2> u = Vector<2>::Constant(2);
	float t = 0.0;
	float h = 0.1;
	KnotPoint<3,2> z2(x,u,t,h);
	EXPECT_TRUE(z2.State().isApprox(Vector<3>::Constant(1)));
	EXPECT_TRUE(z2.Control().isApprox(Vector<2>::Constant(2)));

	KnotPoint<3,2> z3(z);
	EXPECT_TRUE(z3.State().isApprox(Vector<3>::Zero()));
	EXPECT_TRUE(z3.Control().isApprox(Vector<2>::Zero()));

	z3 = z2;
	EXPECT_TRUE(z3.State().isApprox(Vector<3>::Constant(1)));
	EXPECT_TRUE(z3.Control().isApprox(Vector<2>::Constant(2)));

	KnotPoint<3,2> ztmp = KnotPoint<3,2>(3*x,u,t,h);
	KnotPoint<3,2> z4(std::move(ztmp));
	EXPECT_TRUE(z4.State().isApprox(Vector<3>::Constant(3)));
	EXPECT_TRUE(z4.Control().isApprox(Vector<2>::Constant(2)));

	EXPECT_NO_THROW(z4 = std::move(z));
}

TEST(KnotPointTest, DynamicSize) {
	KnotPoint<Eigen::Dynamic, Eigen::Dynamic> z(3,2);
	EXPECT_EQ(z.StateDimension(), 3);
	EXPECT_EQ(z.StateSize(), Eigen::Dynamic);
	EXPECT_EQ(z.ControlDimension(), 2);
	EXPECT_EQ(z.ControlSize(), Eigen::Dynamic);

	EXPECT_TRUE(z.State().isApprox(Vector<3>::Zero()));
	EXPECT_TRUE(z.Control().isApprox(Vector<2>::Zero()));
}

TEST(KnotPointTest, Printing) {
	Vector<3> x = Vector<3>::Constant(1);
	Vector<2> u = Vector<2>::Constant(2);
	float t = 0.0;
	float h = 0.1;
	KnotPoint<3,2> z(x,u,t,h);
	EXPECT_NO_THROW(std::cout << z << std::endl);
}

}  // namespace altro