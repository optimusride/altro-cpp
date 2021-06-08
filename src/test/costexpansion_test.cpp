#include <gtest/gtest.h>
#include "costfunction.hpp"
#include <iostream>

namespace altro {
namespace costexpansion_test {

TEST(CostExpansionTest, Construction)
{
  int n = 3;
  int m = 2; 
  CostExpansion E(n, m);
  Eigen::MatrixXd expansion = E.GetExpansion();
  EXPECT_EQ(expansion.rows(), n+m);
  EXPECT_EQ(expansion.cols(), n+m);
}

TEST(CostExpansionTest, Blocks) 
{
  using Matrix = Eigen::MatrixXd;
  int n = 3;
  int m = 2; 
  CostExpansion E(n, m);
  auto xx = E.dxdx();
  auto xu = E.dxdu();
  auto ux = E.dudx();
  auto uu = E.dudu();
  xu = xu.array() + 1;
  ux = xu.transpose();
  xx = xx.array() - 1;
  uu = uu.array() + 2.5; 
  Matrix expansion2(n+m,n+m);
  
  expansion2 << Matrix::Constant(n,n,-1), Matrix::Constant(n,m,1),
                Matrix::Constant(m,n,1),  Matrix::Constant(m,m,2.5);
  // expansion2 << xx, xu, xu.transpose(), uu;
  std::cout << expansion2 << std::endl;
  std::cout << E.GetExpansion() << std::endl;
  EXPECT_TRUE(E.GetExpansion().isApprox(expansion2));
}

} // namespace costexpansion_test
} // namespace altro