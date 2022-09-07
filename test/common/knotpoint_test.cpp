// Copyright [2021] Optimus Ride Inc.

#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <Eigen/Dense>

#include "altro/common/knotpoint.hpp"
#include "altro/utils/assert.hpp"

namespace altro {

TEST(KnotPointTest, Constructor) {
  KnotPoint<3,2> z;
	EXPECT_TRUE(z.State().isApprox(VectorN<3>::Zero()));
	EXPECT_TRUE(z.Control().isApprox(VectorN<2>::Zero()));

	VectorN<3> x = VectorN<3>::Constant(1);
	VectorN<2> u = VectorN<2>::Constant(2);
	constexpr float t = 0.0;
	constexpr float h = 0.1;
	KnotPoint<3,2> z2(x,u,t,h);
	EXPECT_TRUE(z2.State().isApprox(VectorN<3>::Constant(1)));
	EXPECT_TRUE(z2.Control().isApprox(VectorN<2>::Constant(2)));

	KnotPoint<3,2> z3(z);
	EXPECT_TRUE(z3.State().isApprox(VectorN<3>::Zero()));
	EXPECT_TRUE(z3.Control().isApprox(VectorN<2>::Zero()));

	z3 = z2;
	EXPECT_TRUE(z3.State().isApprox(VectorN<3>::Constant(1)));
	EXPECT_TRUE(z3.Control().isApprox(VectorN<2>::Constant(2)));

	KnotPoint<3,2> ztmp = KnotPoint<3,2>(3*x,u,t,h);
	KnotPoint<3,2> z4(std::move(ztmp));
	EXPECT_TRUE(z4.State().isApprox(VectorN<3>::Constant(3)));
	EXPECT_TRUE(z4.Control().isApprox(VectorN<2>::Constant(2)));

	EXPECT_NO_THROW(z4 = std::move(z));
}

TEST(KnotPointTest, ConstructorDeath) {
  KnotPoint<3,2> z = KnotPoint<3,2>::Random();
  KnotPoint<-1,-1> z3 = KnotPoint<-1,-1>::Random(4,2);

  if (utils::AssertionsActive()) {
    EXPECT_DEATH(z = z3, "Assert.*State sizes must be consistent");
    auto badcopy = [&]() { KnotPoint<3,2> z4(z3); };
    EXPECT_DEATH(badcopy(), "Assert.*State sizes must be consistent");
  
  
    EXPECT_DEATH(z = std::move(z3), "Assert.*State sizes must be consistent");
    auto badmove = [&]() { KnotPoint<3,2> z4(std::move(z3)); };
    EXPECT_DEATH(badmove(), "Assert.*State sizes must be consistent");
  }

	// Copying/moving from static to dynamic is fine
	EXPECT_NO_THROW(z3 = z);
	EXPECT_NO_THROW(z3 = std::move(z));
}

TEST(KnotPointTest, DynamicSize) {
	KnotPoint<Eigen::Dynamic, Eigen::Dynamic> z(3,2);
	EXPECT_EQ(z.StateDimension(), 3);
	EXPECT_EQ(z.StateMemorySize(), Eigen::Dynamic);
	EXPECT_EQ(z.ControlDimension(), 2);
	EXPECT_EQ(z.ControlMemorySize(), Eigen::Dynamic);

	EXPECT_TRUE(z.State().isApprox(VectorN<3>::Zero()));
	EXPECT_TRUE(z.Control().isApprox(VectorN<2>::Zero()));
}

TEST(KnotPointTest, Printing) {
	VectorN<3> x = VectorN<3>::Constant(1);
	VectorN<2> u = VectorN<2>::Constant(2);
	float t = 0.0;
	float h = 0.1;
	KnotPoint<3,2> z(x,u,t,h);
	EXPECT_NO_THROW(std::cout << z << std::endl);
  z.State().setRandom();
  z.State() *= 100;

  Eigen::IOFormat prec(2, 0, ", ", "\n", "", "", "", "");
  fmt::print("{}\n", z.ToString());
}

TEST(KnotPointTest, Random) {
	auto rand_fail = []() { KnotPoint<-1,-1>::Random(); };
  if (utils::AssertionsActive()) {
	  EXPECT_DEATH(rand_fail(), "Assert.*pass in size");
  }

	KnotPoint<-1,-1> rand_dynamic = KnotPoint<-1,-1>::Random(5,3);
	EXPECT_EQ(rand_dynamic.StateDimension(), 5);
	EXPECT_EQ(rand_dynamic.ControlDimension(), 3);
	EXPECT_EQ(rand_dynamic.StateMemorySize(), Eigen::Dynamic);
	EXPECT_LE(std::abs(rand_dynamic.State().maxCoeff()), 1.0);
	EXPECT_LE(std::abs(rand_dynamic.Control().maxCoeff()), 1.0);
	EXPECT_LE(rand_dynamic.GetTime(), 10);
	EXPECT_LE(rand_dynamic.GetStep(), 1);

	KnotPoint<5,3> rand_static = KnotPoint<5,3>::Random();
	EXPECT_EQ(rand_static.StateDimension(), 5);
	EXPECT_EQ(rand_static.ControlDimension(), 3);
	EXPECT_EQ(rand_static.StateMemorySize(), 5);
	EXPECT_LE(std::abs(rand_static.State().maxCoeff()), 1.0);
	EXPECT_LE(std::abs(rand_static.Control().maxCoeff()), 1.0);
	EXPECT_LE(rand_static.GetTime(), 10);
	EXPECT_LE(rand_static.GetStep(), 1);
}

}  // namespace altro