#include <gtest/gtest.h>

#include "trajectory.hpp"

namespace trajectory_test {
using Trajectory = altro::trajectory::Trajectory;
using VectorXd = Eigen::VectorXd;

class TrajectoryTest : public ::testing::Test {
 public:
  int N;
  int n;
  float h;
  std::vector<Eigen::VectorXd> X;
  std::vector<float> steps;
 protected:
  void SetUp() override {
    N = 20;
    n = 5;
    h = 0.1;
    for (int k = 0; k <= N; ++k) {
      X.push_back(Eigen::VectorXd::Random(n));
      if (k < N)
        steps.push_back(h);
    }
  }
};

TEST_F(TrajectoryTest, Construction)
{
  Trajectory Xtraj = Trajectory(X, steps);
  ASSERT_EQ(Xtraj.NumSegments(), N);
  ASSERT_EQ(X.size(), N+1);

  Trajectory Xtraj2 = Trajectory(X, h);
  ASSERT_EQ(Xtraj2.NumSegments(), N);
}

TEST_F(TrajectoryTest, Indexing)
{
  Trajectory Xtraj = Trajectory(X, h);
  VectorXd x0 = Xtraj.GetSample(0);
  EXPECT_TRUE(x0.isApprox(X[0]));

  VectorXd x1 = Xtraj.GetSample(1);
  EXPECT_TRUE(x1.isApprox(X[1]));

  VectorXd xterm = Xtraj.GetSample(N);
  EXPECT_TRUE(xterm.isApprox(X.back()));

  for (int k = 0; k < N; ++k) {
    EXPECT_FLOAT_EQ(Xtraj.GetStep(k), h);
  }
}

TEST_F(TrajectoryTest, IndependentVar) 
{
  Trajectory Xtraj = Trajectory(X, h);
  EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(0), 0);
  EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(N), N*h);

  Xtraj = Trajectory(X, h / 2);
  EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(0), 0);
  EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(N), N*h / 2);

  for (int k = 0; k < N / 2; ++k) {
    steps[k] /= 2;
  }
  Xtraj = Trajectory(X, steps);
  EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(0), 0);
  EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(N), N*h * 3 / 4);
}

} // namespace trajectory_test