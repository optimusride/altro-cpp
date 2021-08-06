#include <gtest/gtest.h>
#include <math.h>
#include <iostream>
#include <chrono>

#include "altro/problem/discretized_model.hpp"
#include "altro/utils/derivative_checker.hpp"
#include "altro/utils/benchmarking.hpp"
#include "examples/unicycle.hpp" 

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
    EXPECT_TRUE(model.CheckHessian(1e-4));
  }
}

TEST(UnicycleTest, BenchmarkRK4) {
  constexpr int NStates = Unicycle::NStates;
  constexpr int NControls = Unicycle::NControls;
  Unicycle model;
  problem::DiscretizedModel<Unicycle> dmodel(model);
  EXPECT_EQ(dmodel.GetIntegrator().StateMemorySize(), 3);
  EXPECT_EQ(dmodel.GetIntegrator().ControlMemorySize(), 2);

  VectorNd<NStates> x = VectorNd<NStates>::Random();
  VectorNd<NControls> u = VectorNd<NControls>::Random();
  VectorNd<NStates> xnext;
  const float t = 1.1;
  const float h = 0.1;

  auto eval = [&]() { dmodel.Evaluate(x, u, t, h, xnext); };
  fmt::print("\nIntegration\n");
  utils::Benchmark<std::chrono::microseconds>(eval, 2000).Print();

  fmt::print("\nJacobian\n");
  MatrixNxMd<NStates, NStates + NControls> jac;
  auto jacobian = [&]() { dmodel.Jacobian(x, u, t, h, jac); };
  utils::Benchmark<std::chrono::microseconds>(jacobian, 2000).Print();
}

}  // namespace examples
}  // namespace altro