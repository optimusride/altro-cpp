// Copyright [2021] Optimus Ride Inc.

#include <gtest/gtest.h>
#include <iostream>

#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/discretized_model.hpp"
#include "examples/quadratic_cost.hpp"
#include "altro/utils/assert.hpp"
#include "examples/triple_integrator.hpp"

namespace altro {
namespace ilqr {

constexpr int dof = 2;
constexpr int n_static = 3 * dof;
constexpr int m_static = dof;
constexpr double tol = 1e-3;

constexpr int HEAP = Eigen::Dynamic;

class KnotPointFunctionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Q = VectorXd::LinSpaced(3 * dof, 1, 3 * dof - 1).asDiagonal();  // NOLINT(readability-magic-numbers)
    R = VectorXd::LinSpaced(dof, 1, dof - 1).asDiagonal();          // NOLINT(readability-magic-numbers)
    H = MatrixXd::Zero(3 * dof, dof);                               // NOLINT(readability-magic-numbers)
    q = VectorXd::LinSpaced(3 * dof, 1, 3 * dof - 1);               // NOLINT(readability-magic-numbers)
    r = VectorXd::LinSpaced(dof, 1, dof - 1);                       // NOLINT(readability-magic-numbers)
    c = 11;                                                         // NOLINT(readability-magic-numbers)
  }

  template <int n_size, int m_size>
  KnotPointFunctions<n_size, m_size> MakeKPF() {
    // Create the model
    using ModelType = problem::DiscretizedModel<examples::TripleIntegrator>;
    examples::TripleIntegrator model(dof);
    ModelType model_d(model);
    std::shared_ptr<ModelType> model_ptr = std::make_shared<ModelType>(model_d);

    // Create the Cost Function
    examples::QuadraticCost costfun(Q, R, H, q, r, c);
    std::shared_ptr<examples::QuadraticCost> costfun_ptr =
        std::make_shared<examples::QuadraticCost>(costfun);

    // Make KPF
    KnotPointFunctions<n_size, m_size> kpf(model_ptr, costfun_ptr);

    return kpf;
  }

  template <int n_size, int m_size>
  void TestGains(double rho = 0) {
    KnotPointFunctions<n_size, m_size> kpf = MakeKPF<n_size, m_size>();
    MatrixXd Qxu = MatrixXd::Random(n, m);
    MatrixXd Quu = MatrixXd::Random(m, m);
    Quu = Quu.transpose() * Quu;
    VectorXd Qu = VectorXd::Random(m);
    kpf.GetActionValueExpansion().dudu() = Quu;
    kpf.GetActionValueExpansion().dxdu() = Qxu;
    kpf.GetActionValueExpansion().du() = Qu;
    kpf.RegularizeActionValue(rho);

    kpf.CalcGains();
    MatrixXd K =
        (Quu + MatrixXd::Identity(m, m) * rho).ldlt().solve(Qxu.transpose());
    kpf.GetFeedbackGain().isApprox(K);
  }

  template <int n_size, int m_size>
  void TestTermCTG() {
    KnotPointFunctions<n_size, m_size> kpf = MakeKPF<n_size, m_size>();
    KnotPoint<n_size, m_size> z = KnotPoint<n_size, m_size>::Random(n, m);

    kpf.CalcCostExpansion(z.State(), z.Control());

    kpf.CalcTerminalCostToGo();
    EXPECT_TRUE(kpf.GetCostToGoHessian().isApprox(Q));
    EXPECT_TRUE(kpf.GetCostToGoGradient().isApprox(Q * z.State() + q));
  }

  template <int n_size, int m_size>
  void TestQExpansion() {
    KnotPointFunctions<n_size, m_size> kpf = MakeKPF<n_size, m_size>();
    KnotPoint<n_size, m_size> z = KnotPoint<n_size, m_size>::Random(n, m);

    MatrixXd Sxx_prev = MatrixXd::Random(n, n);
    Sxx_prev = Sxx_prev.transpose() * Sxx_prev;
    VectorXd Sx_prev = VectorXd::Random(n);

    kpf.CalcCostExpansion(z.State(), z.Control());
    kpf.CalcDynamicsExpansion(z.State(), z.Control(), z.GetTime(), z.GetStep());
    kpf.CalcActionValueExpansion(Sxx_prev, Sx_prev);

    MatrixXd A = kpf.GetDynamicsExpansion().GetA();
    MatrixXd B = kpf.GetDynamicsExpansion().GetB();

    MatrixXd Qxx = Q + A.transpose() * Sxx_prev * A;
    MatrixXd Quu = R + B.transpose() * Sxx_prev * B;
    MatrixXd Qxu = H + A.transpose() * Sxx_prev * B;
    MatrixXd Qx = (Q * z.State() + q) + A.transpose() * Sx_prev;
    MatrixXd Qu = (R * z.Control() + r) + B.transpose() * Sx_prev;

    CostExpansion<n_size, m_size> Q = kpf.GetActionValueExpansion();
    EXPECT_TRUE(Q.dxdx().isApprox(Qxx));
    EXPECT_TRUE(Q.dxdu().isApprox(Qxu));
    EXPECT_TRUE(Q.dudu().isApprox(Quu));
    EXPECT_TRUE(Q.dx().isApprox(Qx));
    EXPECT_TRUE(Q.du().isApprox(Qu));
  }

  template <int n_size, int m_size>
  void TestCTG() {
    MatrixXd Q = MatrixXd::Random(n + m, n + m);
    Q = Q.transpose() * Q;
    MatrixXd Qxx = Q.topLeftCorner(n, n);
    MatrixXd Qxu = Q.topRightCorner(n, m);
    MatrixXd Qux = Q.bottomLeftCorner(m, n);
    MatrixXd Quu = Q.bottomRightCorner(m, m);
    VectorXd Qx = VectorXd::Random(n);
    VectorXd Qu = VectorXd::Random(m);

    MatrixXd K = MatrixXd::Random(m, n);
    VectorXd d = VectorXd::Random(m);
    ASSERT_TRUE(Qxu.transpose().isApprox(Qux));

    MatrixXd Sxx =
        Qxx + K.transpose() * Quu * K + K.transpose() * Qux + Qxu * K;
    MatrixXd Sx = Qx + K.transpose() * Quu * d + K.transpose() * Qu + Qxu * d;
    double deltaV = d.dot(Qu) + 0.5 * d.dot(Quu * d);  // NOLINT(readability-magic-numbers)

    KnotPointFunctions<n_size, m_size> kpf = this->MakeKPF<n_size, m_size>();
    kpf.GetActionValueExpansion().dxdx() = Qxx;
    kpf.GetActionValueExpansion().dxdu() = Qxu;
    kpf.GetActionValueExpansion().dudu() = Quu;
    kpf.GetActionValueExpansion().dx() = Qx;
    kpf.GetActionValueExpansion().du() = Qu;
    kpf.GetFeedbackGain() = K;
    kpf.GetFeedforwardGain() = d;

    kpf.CalcCostToGo();
    EXPECT_TRUE(kpf.GetCostToGoHessian().isApprox(Sxx));
    EXPECT_TRUE(kpf.GetCostToGoGradient().isApprox(Sx));
    EXPECT_DOUBLE_EQ(kpf.GetCostToGoDelta(), deltaV);
  }

  int n = 3 * dof;
  int m = dof;
  MatrixXd Q;
  MatrixXd R;
  MatrixXd H;
  VectorXd q;
  VectorXd r;
  double c;
};

TEST_F(KnotPointFunctionsTest, Construction) {
  // int n = n_static;
  // int m = m_static;
  auto create_static = [&]() { this->MakeKPF<n_static, m_static>(); };
  auto create_dynamic = [&]() { this->MakeKPF<HEAP, HEAP>(); };
  EXPECT_NO_THROW(create_static());
  EXPECT_NO_THROW(create_dynamic());
}

TEST_F(KnotPointFunctionsTest, ConstructionDeath) {
  // Create the model
  using ModelType = problem::DiscretizedModel<examples::TripleIntegrator>;
  examples::TripleIntegrator model(dof);
  ModelType model_d(model);
  std::shared_ptr<ModelType> model_ptr = std::make_shared<ModelType>(model_d);

  // Create the Cost Function
  examples::QuadraticCost costfun(Q, R, H, q, r, c);
  std::shared_ptr<examples::QuadraticCost> costfun_ptr =
      std::make_shared<examples::QuadraticCost>(costfun);

  if (utils::AssertionsActive()) {
    auto null_dynamics = [&]() {
      KnotPointFunctions<n_static, m_static> kpf(nullptr, costfun_ptr);
    };
    EXPECT_DEATH(null_dynamics(), "Assert.*null dynamics pointer");
  
    auto null_costfun = [&]() {
      KnotPointFunctions<n_static, m_static> kpf(model_ptr, nullptr);
    };
    EXPECT_DEATH(null_costfun(), "Assert.*null cost function pointer");
  }
}

TEST_F(KnotPointFunctionsTest, Cost) {
  // Create random data
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random();
  KnotPoint<HEAP, HEAP> z_dynamic(z_static);

  // Extract out states and controls
  VectorN<n_static> x_static = z_static.State();
  VectorN<m_static> u_static = z_static.Control();
  VectorXd x = z_dynamic.State();
  VectorXd u = z_dynamic.Control();
  EXPECT_TRUE(x.isApprox(x_static));
  EXPECT_TRUE(u.isApprox(u_static));

  // Make the KnotPointFunctions types
  KnotPointFunctions<n_static, m_static> kpf_static =
      MakeKPF<n_static, m_static>();
  KnotPointFunctions<HEAP, HEAP> kpf_dynamic = MakeKPF<HEAP, HEAP>();

  // Compute the cost
  double J_static = kpf_static.Cost(x_static, u_static);
  double J_dynamic = kpf_dynamic.Cost(x, u);
  double J_expected =
      0.5 * (x.dot(Q * x) + u.dot(R * u)) + q.dot(x) + r.dot(u) + c;

  // Compare
  EXPECT_DOUBLE_EQ(J_dynamic, J_static);
  EXPECT_DOUBLE_EQ(J_dynamic, J_expected);
  EXPECT_DOUBLE_EQ(J_static, J_expected);

  // Mix up static and dynamic
  EXPECT_DOUBLE_EQ(kpf_static.Cost(x, u), J_expected);
  EXPECT_DOUBLE_EQ(kpf_dynamic.Cost(x_static, u_static), J_expected);
}

TEST_F(KnotPointFunctionsTest, Dynamics) {
  // Create random data
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random();
  KnotPoint<HEAP, HEAP> z_dynamic(z_static);

  // Extract out states and controls
  VectorN<n_static> x_static = z_static.State();
  VectorN<m_static> u_static = z_static.Control();
  VectorXd x = z_dynamic.State();
  VectorXd u = z_dynamic.Control();
  float t = z_static.GetTime();
  float h = z_static.GetStep();
  EXPECT_TRUE(x.isApprox(x_static));
  EXPECT_TRUE(u.isApprox(u_static));

  // Make the KnotPointFunctions types
  KnotPointFunctions<n_static, m_static> kpf_static =
      MakeKPF<n_static, m_static>();
  KnotPointFunctions<HEAP, HEAP> kpf_dynamic = MakeKPF<HEAP, HEAP>();

  // Evaluate the dynamics
  VectorXd xnext(n);
  VectorN<n_static> xnext_static;
  kpf_static.Dynamics(x_static, u_static, t, h, xnext_static);
  kpf_dynamic.Dynamics(x, u, t, h, xnext);
  EXPECT_TRUE(xnext.isApprox(xnext_static));

  // Mix up static and dynamic
  kpf_dynamic.Dynamics(x_static, u_static, t, h, xnext);
  EXPECT_TRUE(xnext.isApprox(xnext_static));
  kpf_dynamic.Dynamics(x_static, u_static, t, h, xnext_static);
  EXPECT_TRUE(xnext.isApprox(xnext_static));
  kpf_static.Dynamics(x, u, t, h, xnext);
  EXPECT_TRUE(xnext.isApprox(xnext_static));
  kpf_static.Dynamics(x, u, t, h, xnext_static);
  EXPECT_TRUE(xnext.isApprox(xnext_static));
  kpf_static.Dynamics(x, u_static, t, h, xnext);
  EXPECT_TRUE(xnext.isApprox(xnext_static));
}

TEST_F(KnotPointFunctionsTest, CostExpansion) {
  // Create random data
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random();
  KnotPoint<HEAP, HEAP> z_dynamic(z_static);

  // Extract out states and controls
  VectorN<n_static> x_static = z_static.State();
  VectorN<m_static> u_static = z_static.Control();
  VectorXd x = z_dynamic.State();
  VectorXd u = z_dynamic.Control();
  EXPECT_TRUE(x.isApprox(x_static));
  EXPECT_TRUE(u.isApprox(u_static));

  // Make the KnotPointFunctions types
  KnotPointFunctions<n_static, m_static> kpf_static =
      MakeKPF<n_static, m_static>();
  KnotPointFunctions<HEAP, HEAP> kpf_dynamic = MakeKPF<HEAP, HEAP>();

  // Calculate the cost expansion
  kpf_static.CalcCostExpansion(x, u);
  kpf_dynamic.CalcCostExpansion(x, u);

  // Extract expansions and check
  CostExpansion<n_static, m_static> cost_expansion =
      kpf_static.GetCostExpansion();
  EXPECT_TRUE(cost_expansion.du().isApprox(R * u + r));
  EXPECT_TRUE(cost_expansion.dx().isApprox(Q * x + q));
  EXPECT_TRUE(cost_expansion.dxdx().isApprox(Q));
  EXPECT_TRUE(cost_expansion.dudu().isApprox(R));
  EXPECT_TRUE(cost_expansion.dxdu().isApprox(H));

  CostExpansion<HEAP, HEAP> cost_expansion2 = kpf_dynamic.GetCostExpansion();
  EXPECT_TRUE(cost_expansion2.du().isApprox(R * u + r));
  EXPECT_TRUE(cost_expansion2.dx().isApprox(Q * x + q));
  EXPECT_TRUE(cost_expansion2.dxdx().isApprox(Q));
  EXPECT_TRUE(cost_expansion2.dudu().isApprox(R));
  EXPECT_TRUE(cost_expansion2.dxdu().isApprox(H));
}

TEST_F(KnotPointFunctionsTest, DynamicsExpansion) {
  // Create random data
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random();
  KnotPoint<HEAP, HEAP> z_dynamic(z_static);

  // Extract out states and controls
  VectorN<n_static> x_static = z_static.State();
  VectorN<m_static> u_static = z_static.Control();
  VectorXd x = z_dynamic.State();
  VectorXd u = z_dynamic.Control();
  float t = z_static.GetTime();
  float h = z_static.GetStep();
  EXPECT_TRUE(x.isApprox(x_static));
  EXPECT_TRUE(u.isApprox(u_static));

  // Make the KnotPointFunctions types
  KnotPointFunctions<n_static, m_static> kpf_static =
      MakeKPF<n_static, m_static>();
  KnotPointFunctions<HEAP, HEAP> kpf_dynamic = MakeKPF<HEAP, HEAP>();

  // Calculate dynamics expansion
  kpf_static.CalcDynamicsExpansion(x_static, u_static, t, h);
  kpf_dynamic.CalcDynamicsExpansion(x, u, t, h);

  DynamicsExpansion<n_static, m_static> dyn_exp =
      kpf_static.GetDynamicsExpansion();
  MatrixXd jac(n, n + m);
  kpf_static.GetModelPtr()->Jacobian(x, u, t, h, jac);
  EXPECT_TRUE(dyn_exp.GetJacobian().isApprox(jac));

  DynamicsExpansion<HEAP, HEAP> dyn_exp2 = kpf_dynamic.GetDynamicsExpansion();
  EXPECT_TRUE(dyn_exp2.GetJacobian().isApprox(jac));
}

TEST_F(KnotPointFunctionsTest, TerminalCostToGoStatic) {
  TestTermCTG<n_static, m_static>();
}

TEST_F(KnotPointFunctionsTest, TerminalCostToGoDynamic) {
  TestTermCTG<HEAP, HEAP>();
}

TEST_F(KnotPointFunctionsTest, ActionValueExpansionStatic) {
  TestQExpansion<n_static, m_static>();
}

TEST_F(KnotPointFunctionsTest, ActionValueExpansionDynamic) {
  TestQExpansion<HEAP, HEAP>();
}

TEST_F(KnotPointFunctionsTest, CalcGainsStatic) {
  TestGains<n_static, m_static>();
  TestGains<n_static, m_static>(tol);
}

TEST_F(KnotPointFunctionsTest, CalcGainsDynamic) { 
  TestGains<HEAP, HEAP>(); 
  TestGains<HEAP, HEAP>(tol);
}

TEST_F(KnotPointFunctionsTest, CalcCostToGoStatic) {
  TestCTG<n_static, m_static>();
}

TEST_F(KnotPointFunctionsTest, CalcCostToGoDynamic) { TestCTG<HEAP, HEAP>(); }

}  // namespace ilqr
}  // namespace altro