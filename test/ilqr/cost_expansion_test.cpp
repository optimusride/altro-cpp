#include <gtest/gtest.h>
#include <iostream>

#include "ilqr/cost_expansion.hpp"
#include "quadratic_cost.hpp"

namespace altro {
namespace ilqr {
class CostExpansionTest : public ::testing::Test 
{
 public:
	int n = 4;
	int m = 3;
	MatrixXd Q = Eigen::Vector4d(1,2,3,4).asDiagonal();
	MatrixXd R = Eigen::Vector3d(5,6,7).asDiagonal();
	MatrixXd H = Eigen::MatrixXd::Zero(4,3);
	VectorXd q = Eigen::Vector4d::LinSpaced(4,8,11);
	VectorXd r = Eigen::Vector3d::LinSpaced(3,12,14);
	double c = 1.0;
 protected:
	void SetUp() {
	}
};

TEST_F(CostExpansionTest, Construction) {
	CostExpansion<4,3> expansion(n,m);
	int a = 1;
	std::cout << a << std::endl;
}

}  // namespace ilqr
}  // namespace altro