#include <gtest/gtest.h>
#include <iostream>

#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/discretized_model.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/triple_integrator.hpp"

namespace altro {
namespace ilqr {

constexpr int dof = 2;
constexpr int n_static = 3 * dof;
constexpr int m_static = dof;

constexpr int HEAP = Eigen::Dynamic;

class KnotPointFunctionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Q = VectorXd::LinSpaced(3 * dof, 1, 3 * dof - 1).asDiagonal();
    R = VectorXd::LinSpaced(dof, 1, dof - 1).asDiagonal();
    H = MatrixXd::Zero(3 * dof, dof);
    q = VectorXd::LinSpaced(3 * dof, 1, 3 * dof - 1);
    r = VectorXd::LinSpaced(dof, 1, dof - 1);
    c = 11;
  }

  template <int n, int m>
  KnotPointFunctions<n, m> MakeKPF() {
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
    KnotPointFunctions<n, m> kpf(model_ptr, costfun_ptr);

    return kpf;
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

  auto null_dynamics = [&]() {
    KnotPointFunctions<n_static, m_static> kpf(nullptr, costfun_ptr);
  };
  EXPECT_DEATH(null_dynamics(), "Assert.*null dynamics pointer");

  auto null_costfun = [&]() {
    KnotPointFunctions<n_static, m_static> kpf(model_ptr, nullptr);
  };
  EXPECT_DEATH(null_costfun(), "Assert.*null cost function pointer");
}

TEST_F(KnotPointFunctionsTest, Cost) {
  // Create random data
  KnotPoint<n_static, m_static> z_static =
      KnotPoint<n_static, m_static>::Random();
  KnotPoint<HEAP, HEAP> z_dynamic(z_static);

  // Extract out states and controls
  Vector<n_static> x_static = z_static.State();
  Vector<m_static> u_static = z_static.Control();
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
  Vector<n_static> x_static = z_static.State();
  Vector<m_static> u_static = z_static.Control();
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
  Vector<n_static> xnext_static;
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
  Vector<n_static> x_static = z_static.State();
  Vector<m_static> u_static = z_static.Control();
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
  Vector<n_static> x_static = z_static.State();
  Vector<m_static> u_static = z_static.Control();
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

}  // namespace ilqr
}  // namespace altro