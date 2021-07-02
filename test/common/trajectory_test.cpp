#include <gtest/gtest.h>

#include "altro/common/trajectory.hpp"
#include "altro/utils/assert.hpp"

namespace altro {

class TrajectoryTest : public ::testing::Test 
{
  public:
    int N = 10;
    int n = 3;
    int m = 2;
    float h = 0.1;
    std::vector<VectorN<3>> X;
    std::vector<VectorN<2>> U;
    std::vector<KnotPoint<3,2>> knotpoints;
    std::vector<float> times;
  protected:
    void SetUp() override {
      for (int k = 0; k <= N; ++k) {
        X.push_back(VectorN<3>::Constant(k));
        if (k < N) {
          U.push_back(VectorN<2>::Constant(N+k));
        }
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
  EXPECT_TRUE(traj2.CheckTimeConsistency());
  EXPECT_TRUE(traj3.CheckTimeConsistency());
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
      knotpoints2.emplace_back(x, u*0, h*k, 0.0F);
    }
  }

  TrajectoryXXd traj2(knotpoints2);
  EXPECT_EQ(traj2.StateDimension(0), n);
  EXPECT_EQ(traj2.ControlDimension(0), m);
  EXPECT_EQ(traj2.StateDimension(N-1), n-1);
  EXPECT_EQ(traj2.ControlDimension(N-1), m-1);
}

TEST_F(TrajectoryTest, SetStep) {
  Trajectory<3,2> traj(knotpoints);
  EXPECT_TRUE(traj.CheckTimeConsistency());
  traj[1].SetTime(2 * h);
  EXPECT_FALSE(traj.CheckTimeConsistency());
  EXPECT_FLOAT_EQ(traj[1].GetTime(), 2 * h);

  traj.SetUniformStep(2 * h);
  EXPECT_FLOAT_EQ(traj.GetTime(N), 2 * N * h);
  EXPECT_FLOAT_EQ(traj.GetStep(N-1), 2 * h);
  EXPECT_FLOAT_EQ(traj.GetStep(N), 0);
  EXPECT_TRUE(traj.CheckTimeConsistency());
}

TEST_F(TrajectoryTest, Iteration) {
  Trajectory<3,2> traj(knotpoints);
  std::vector<KnotPoint<3,2>>::iterator z_itr = traj.begin();
  EXPECT_TRUE(z_itr->State().isApprox(X[0]));
  ++z_itr;   
  EXPECT_TRUE(z_itr->State().isApprox(X[1]));
  --z_itr;
  EXPECT_TRUE(z_itr->State().isApprox(X[0]));

  std::vector<KnotPoint<3,2>>::iterator z_itr_end = traj.end();
  EXPECT_TRUE((--z_itr_end)->State().isApprox(X[N]));

  int k = 0;
  for (auto z = traj.begin(); z != traj.end(); ++z, ++k) {
    EXPECT_TRUE(z->State().isApprox(X[k]));
  }
  EXPECT_EQ(k, N+1);

  k = 0;
  for (const auto z : traj) {
    EXPECT_TRUE(z.State().isApprox(X[k]));
    ++k;
  }
  EXPECT_EQ(k, N+1);
}

TEST_F(TrajectoryTest, Copy) {
  Trajectory<3,2> traj(knotpoints);
  Trajectory<3,2> traj2(traj);
  for (int k = 0; k < N; ++k) {
    EXPECT_TRUE(traj[k].GetStateControl().isApprox(traj2[k].GetStateControl()));
    EXPECT_FLOAT_EQ(traj[k].GetTime(), traj2[k].GetTime());
    EXPECT_FLOAT_EQ(traj[k].GetStep(), traj2[k].GetStep());
  }
  traj.State(0).setConstant(5.0);
  EXPECT_FALSE(traj.State(0).isApprox(traj2.State(0)));
}

TEST_F(TrajectoryTest, SetZero) {
  Trajectory<3,2> traj(knotpoints);
  traj.SetZero();
  for (auto z : traj) {
    EXPECT_FLOAT_EQ(z.GetStateControl().norm(), 0);
  }
  EXPECT_FLOAT_EQ(traj.GetStep(0), h);
  EXPECT_TRUE(traj.CheckTimeConsistency());
}

}  // namespace altro
