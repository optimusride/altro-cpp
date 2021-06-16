#include <gtest/gtest.h>

#include "trajectory.hpp"
#include <iostream>

namespace altro {

class TrajectoryTest : public ::testing::Test 
{
  public:
    int N = 10;
    int n = 3;
    int m = 2;
    float h = 0.1;
    std::vector<Vector<3>> X;
    std::vector<Vector<2>> U;
    std::vector<KnotPoint<3,2>> knotpoints;
    std::vector<float> times;
  protected:
    void SetUp() override {
      for (int k = 0; k <= N; ++k) {
        X.push_back(Vector<3>::Constant(k));
        if (k < N)
          U.push_back(Vector<2>::Constant(N+k));
        times.push_back(k*h);
      }
      for (int k = 0; k < N; ++k) {
        knotpoints.emplace_back(X[k], U[k], k*h, h);
      }
      knotpoints.emplace_back(X[N], 0*U[N], N*h, 0);
    }
};

TEST_F(TrajectoryTest, Constructor) {
  Trajectory<3,2> traj(N);
  Trajectory<3,2> traj2(knotpoints);
  Trajectory<3,2> traj3(X, U, times);

  for (int k = 0; k < N; ++k) {
    EXPECT_FLOAT_EQ(traj.State(k).norm(), 0);
    EXPECT_FLOAT_EQ(traj.Control(k).norm(), 0);
    EXPECT_TRUE(traj2.State(k).isApprox(X[k]));
    EXPECT_TRUE(traj2.Control(k).isApprox(U[k]));
    EXPECT_TRUE(traj3.State(k).isApprox(X[k]));
    EXPECT_TRUE(traj3.Control(k).isApprox(U[k]));
  }
  EXPECT_TRUE(traj2.CheckTimes());
  EXPECT_TRUE(traj3.CheckTimes());
}

TEST_F(TrajectoryTest, DynamicSize) {
  TrajectoryXXd traj(n,m,N);

  for (int k = 0; k <= N; ++k) {
    EXPECT_EQ(n, traj.StateDimension(k));
    EXPECT_EQ(m, traj.ControlDimension(k));
    EXPECT_FLOAT_EQ(traj.State(k).norm(), 0);
    EXPECT_FLOAT_EQ(traj.Control(k).norm(), 0);
  }

  std::vector<KnotPoint<Eigen::Dynamic,Eigen::Dynamic>> knotpoints2;
  for (int k = 0; k <= N; ++k) {
    int n2 = n;
    int m2 = m;
    if (k > N / 2) {
      --n2;
      --m2;
    }
    VectorXd x = VectorXd::Constant(n2, k);
    VectorXd u = VectorXd::Constant(m2, k);
    if (k < N) {
      knotpoints2.emplace_back(x, u, h*k, h);
    } else {
      knotpoints2.emplace_back(x, u*0, h*k, 0.0f);
    }
  }

  TrajectoryXXd traj2(knotpoints2);
  EXPECT_EQ(traj2.StateDimension(0), n);
  EXPECT_EQ(traj2.ControlDimension(0), m);
  EXPECT_EQ(traj2.StateDimension(N-1), n-1);
  EXPECT_EQ(traj2.ControlDimension(N-1), m-1);
}

} // namespace altro


// namespace trajectory_test {

// using Trajectory = altro::trajectory::Trajectory;
// using VectorXd = Eigen::VectorXd;

// class TrajectoryTest : public ::testing::Test {
//  public:
//   int N;
//   int n;
//   float h;
//   std::vector<Eigen::VectorXd> X;
//   std::vector<float> steps;
//  protected:
//   void SetUp() override {
//     N = 20;
//     n = 5;
//     h = 0.1;
//     for (int k = 0; k <= N; ++k) {
//       X.push_back(Eigen::VectorXd::Random(n));
//       if (k < N)
//         steps.push_back(h);
//     }
//   }
// };

// TEST_F(TrajectoryTest, Construction)
// {
//   Trajectory Xtraj = Trajectory(X, steps);
//   ASSERT_EQ(Xtraj.NumSegments(), N);
//   ASSERT_EQ(X.size(), N+1);

//   Trajectory Xtraj2 = Trajectory(X, h);
//   ASSERT_EQ(Xtraj2.NumSegments(), N);
// }

// TEST_F(TrajectoryTest, Indexing)
// {
//   Trajectory Xtraj = Trajectory(X, h);
//   VectorXd x0 = Xtraj.GetSample(0);
//   EXPECT_TRUE(x0.isApprox(X[0]));

//   VectorXd x1 = Xtraj.GetSample(1);
//   EXPECT_TRUE(x1.isApprox(X[1]));

//   VectorXd xterm = Xtraj.GetSample(N);
//   EXPECT_TRUE(xterm.isApprox(X.back()));

//   for (int k = 0; k < N; ++k) {
//     EXPECT_FLOAT_EQ(Xtraj.GetStep(k), h);
//   }
// }

// TEST_F(TrajectoryTest, IndependentVar) 
// {
//   Trajectory Xtraj = Trajectory(X, h);
//   EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(0), 0);
//   EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(N), N*h);

//   Xtraj = Trajectory(X, h / 2);
//   EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(0), 0);
//   EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(N), N*h / 2);

//   for (int k = 0; k < N / 2; ++k) {
//     steps[k] /= 2;
//   }
//   Xtraj = Trajectory(X, steps);
//   EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(0), 0);
//   EXPECT_FLOAT_EQ(Xtraj.GetIndependentVar(N), N*h * 3 / 4);
// }

// } // namespace trajectory_test