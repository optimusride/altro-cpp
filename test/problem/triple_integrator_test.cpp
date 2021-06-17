#include <gtest/gtest.h>

#include "utils/derivative_checker.hpp"
#include "triple_integrator.hpp" 
#include "problem/discretized_model.hpp"

namespace altro {
namespace examples {

TEST(TripleIntegratorTest, Constructor) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  EXPECT_EQ(3 * degrees_of_freedom, model2.StateDimension());
  EXPECT_EQ(degrees_of_freedom, model2.ControlDimension());

  TripleIntegrator model;
  degrees_of_freedom = 1;
  EXPECT_EQ(3 * degrees_of_freedom, model.StateDimension());
  EXPECT_EQ(degrees_of_freedom, model.ControlDimension());
}

TEST(TripleIntegratorTest, ConstructorDeath) {
  EXPECT_DEATH(TripleIntegrator model_fail = TripleIntegrator(0),
               "Assert.*greater than 0");
}

TEST(TripleIntegratorTest, Evaluate) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  VectorXd x = VectorXd::Random(model2.StateDimension());
  VectorXd u = VectorXd::Random(model2.ControlDimension());
  VectorXd xdot = model2.Evaluate(x, u, 0.0);

  VectorXd xdot_ans(model2.StateDimension());
  xdot_ans << x.tail(2 * degrees_of_freedom), u;
  EXPECT_TRUE(xdot_ans.isApprox(xdot));

  // Call it as a functor
  VectorXd xdot2 = model2(x, u, 1.0f);
  EXPECT_TRUE(xdot_ans.isApprox(xdot2));
}

TEST(TripleIntegratorTest, Jacobian) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  int n = model2.StateDimension();
  int m = model2.ControlDimension();
  double t = 0.0;
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  MatrixXd jac = MatrixXd::Zero(n, n + m);
  model2.Jacobian(x, u, t, jac);
  MatrixXd jac_ans(n, n + m);
  // clang-format off
  jac_ans << 0,0, 1,0, 0,0, 0,0,
             0,0, 0,1, 0,0, 0,0,
             0,0, 0,0, 1,0, 0,0,
             0,0, 0,0, 0,1, 0,0,
             0,0, 0,0, 0,0, 1,0,
             0,0, 0,0, 0,0, 0,1;
  // clang-format on
  EXPECT_TRUE(jac_ans.isApprox(jac));
}

TEST(TripleIntegrator, Hessian) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  int n = model2.StateDimension();
  int m = model2.ControlDimension();
  double t = 0.0;
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  VectorXd b = VectorXd::Random(n);
  MatrixXd hess = MatrixXd::Random(n, n);
  model2.Hessian(x, u, t, b, hess);
  EXPECT_TRUE(hess.isApproxToConstant(0.0));
}

TEST(TripleIntegrator, Discretize) {
  int degrees_of_freedom = 2;
  TripleIntegrator model_cont(degrees_of_freedom);
  problem::DiscretizedModel<TripleIntegrator> model_discrete(model_cont);
  int n = model_discrete.StateDimension();
  int m = model_discrete.ControlDimension();
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  float t = 1.1;
  float h = 0.1;
  VectorXd xnext = model_discrete.Evaluate(x, u, t, h);
  VectorXd xdot = model_cont.Evaluate(x, u, t);
  EXPECT_GT((xdot - xnext).norm(), 1e-6);
  VectorXd k1 = model_cont.Evaluate(x, u, t) * h;
  VectorXd k2 = model_cont.Evaluate(x + k1 * 0.5, u, t) * h;
  VectorXd k3 = model_cont.Evaluate(x + k2 * 0.5, u, t) * h;
  VectorXd k4 = model_cont.Evaluate(x + k3, u, t) * h;
  VectorXd xnext2 = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
  EXPECT_TRUE(xnext.isApprox(xnext2));

  VectorXd xnext3 = model_discrete(x, u, t, h);
  EXPECT_TRUE(xnext.isApprox(xnext3));

  MatrixXd jac = MatrixXd::Zero(n, n + m);
  model_discrete.Jacobian(x, u, t, h, jac);

  MatrixXd jac_cont = MatrixXd::Zero(n, n + m);
  model_cont.Jacobian(x, u, t, jac_cont);
  MatrixXd A = jac_cont.topLeftCorner(n, n);
  MatrixXd B = jac_cont.topRightCorner(n, m);
  MatrixXd K1 = A * h;
  MatrixXd K2 = A * h + 0.5 * A * A * pow(h, 2);
  MatrixXd K3 = A * h + 0.5 * A * A * pow(h, 2) + 0.25 * A * A * A * pow(h, 3);
  MatrixXd K4 = A * h + A * A * pow(h, 2) + 0.5 * A * A * A * pow(h, 3) +
                0.25 * A * A * A * A * pow(h, 4);
  MatrixXd Ad = MatrixXd::Identity(n, n) + (K1 + 2 * K2 + 2 * K3 + K4) / 6;

  MatrixXd B1 = B * h;
  MatrixXd B2 = B * h + 0.5 * A * B * h * h;
  MatrixXd B3 = B * h + 0.5 * A * B * h * h + 0.25 * A * A * B * h * h * h;
  MatrixXd B4 = B * h + A * B * h * h + 0.5 * A * A * B * h * h * h +
                0.25 * A * A * A * B * h * h * h * h;
  MatrixXd Bd = (B1 + 2 * B2 + 2 * B3 + B4) / 6;
  EXPECT_TRUE(jac.topLeftCorner(n, n).isApprox(Ad));
  EXPECT_TRUE(jac.topRightCorner(n, m).isApprox(Bd));
}

TEST(TripleIntegratorTest, EulerIntegration) {
  int degrees_of_freedom = 2;
  TripleIntegrator model_cont(degrees_of_freedom);
  problem::DiscretizedModel<TripleIntegrator, problem::ExplicitEuler> model_discrete(model_cont);
  int n = model_discrete.StateDimension();
  int m = model_discrete.ControlDimension();
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  float t = 1.1;
  float h = 0.1;
  VectorXd xnext = model_discrete.Evaluate(x, u, t, h);
  EXPECT_TRUE(xnext.isApprox(x + model_cont(x, u, t) * h));

  MatrixXd jac = MatrixXd::Zero(n, n + m);
  model_discrete.Jacobian(x, u, t, h, jac);
  MatrixXd jac_ans(n, n + m);
  jac_ans << 1, 0, h, 0, 0, 0, 0, 0, 0, 1, 0, h, 0, 0, 0, 0, 0, 0, 1, 0, h, 0,
      0, 0, 0, 0, 0, 1, 0, h, 0, 0, 0, 0, 0, 0, 1, 0, h, 0, 0, 0, 0, 0, 0, 1, 0,
      h;
  EXPECT_TRUE(jac.isApprox(jac_ans));
}

TEST(TripleIntegratorTest, DerivativeChecks) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  int n = model2.StateDimension();
  int m = model2.ControlDimension();
  double t = 0.0;
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  MatrixXd jac = MatrixXd::Zero(n, n + m);
  model2.Jacobian(x, u, t, jac);
  MatrixXd A = jac.topLeftCorner(n, n);
  MatrixXd B = jac.topRightCorner(n, m);

  VectorXd z(n + m);
  z << x, u;

  auto fx = [&](auto x_) { return model2(x_, u, t); };
  auto fd_A = utils::FiniteDiffJacobian<-1,-1>(fx, x);
  EXPECT_TRUE(fd_A.isApprox(A, 1e-6));

  auto fu = [&](auto u_) { return model2(x, u_, t); };
  auto fd_B = utils::FiniteDiffJacobian<-1,-1>(fu, u);
  EXPECT_TRUE(fd_B.isApprox(B, 1e-6));

  auto fz = [&](auto z) { return model2(z.head(n), z.tail(m), t); };
  auto fd_jac = utils::FiniteDiffJacobian<-1,-1>(fz, z);
  EXPECT_TRUE(fd_jac.isApprox(jac, 1e-6));

  VectorXd b = VectorXd::Random(n);
  auto jvp = [&](auto z) -> MatrixXd {
    MatrixXd jac_(n, n + m);
    model2.Jacobian(z.head(n), z.tail(m), t, jac_);
    return jac_.transpose() * b;
  };
  auto hess = utils::FiniteDiffJacobian<-1,-1>(jvp, z);
  EXPECT_FLOAT_EQ(hess.norm(), 0.0);

  EXPECT_TRUE(model2.CheckJacobian(x, u, t));

  for (int i = 0; i < 100; ++i) EXPECT_TRUE(model2.CheckJacobian());
}

TEST(TripleIntegratorTest, DiscreteDerivativeChecks) {
  int degrees_of_freedom = 2;
  TripleIntegrator model_cont(degrees_of_freedom);
  problem::DiscretizedModel<TripleIntegrator> model_discrete(model_cont);

  for (int i = 0; i < 100; ++i) EXPECT_TRUE(model_discrete.CheckJacobian());
}

TEST(TripleIntegratorTest, HessianChecks) {
  int degrees_of_freedom = 2;
  TripleIntegrator model_cont(degrees_of_freedom);
  problem::DiscretizedModel<TripleIntegrator> model_discrete(model_cont);

  for (int i = 0; i < 10; ++i) EXPECT_TRUE(model_cont.CheckHessian());
  // TODO(bjackson) [SW-14571] add second-order RK4 derivatives
  // for (int i = 0; i < 10; ++i) EXPECT_TRUE(model_discrete.CheckHessian());
}

}  // namespace dynamics_examples
}  // namespace altro
