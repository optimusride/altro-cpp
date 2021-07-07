#include <gtest/gtest.h>
#include <math.h>
#include <iostream>

#include "altro/utils/derivative_checker.hpp"
#include "examples/unicycle.hpp" 
#include "altro/problem/discretized_model.hpp"

namespace altro {
namespace examples {

TEST(UnicycleTest, Constructor) {
	Unicycle model;
	EXPECT_EQ(3, model.StateDimension());
	EXPECT_EQ(2, model.ControlDimension());
	EXPECT_TRUE(model.HasHessian());
}

TEST(UnicycleTest, Evaluate) {
	Unicycle model;
	double px = 1.0;
	double py = 2.0;
	double theta = M_PI / 3;
	double v = 0.1;
	double w = -0.3;
	VectorXd x = Eigen::Vector3d(px, py, theta);
	VectorXd u = Eigen::Vector2d(v, w);
	float t = 0.1;
	VectorXd xdot = model(x, u, t);
	VectorXd xdot_expected = Eigen::Vector3d(v * 0.5, v * sqrt(3) / 2.0, w);
	EXPECT_TRUE(xdot.isApprox(xdot_expected));
}

TEST(UnicycleTest, CheckJacobian) {
	Unicycle model;
	for (int i = 0; i < 10; ++i) {
		EXPECT_TRUE(model.CheckJacobian());
	}
}

TEST(UnicycleTest, CheckHessian) {
	Unicycle model;
	for (int i = 0; i < 10; ++i) {
		EXPECT_TRUE(model.CheckHessian(1e-4, true));
	}
}

}  // namespace examples
}  // namespace altro