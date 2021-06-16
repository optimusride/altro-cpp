#include <gtest/gtest.h>
#include "triple_integrator.hpp" 
#include "discretized_model.hpp"

namespace altro {
namespace examples {

TEST(TripleIntegratorTest, Constructor) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  EXPECT_EQ(3*degrees_of_freedom, model2.StateDimension());
  EXPECT_EQ(degrees_of_freedom, model2.ControlDimension());

  TripleIntegrator model;
  degrees_of_freedom = 1;
  EXPECT_EQ(3*degrees_of_freedom, model.StateDimension());
  EXPECT_EQ(degrees_of_freedom, model.ControlDimension());

  EXPECT_TRUE(model.HasHessian());
}

TEST(TripleIntegrator, ConstructorDeath) {
  EXPECT_DEATH(TripleIntegrator model_fail = TripleIntegrator(0), "Assert.*greater than 0");
}

TEST(TripleIntegrator, Evaluate)
{
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  VectorXd x = VectorXd::Random(model2.StateDimension());
  VectorXd u = VectorXd::Random(model2.ControlDimension());
  VectorXd xdot = model2.Evaluate(x, u, 0.0);

  VectorXd xdot_ans(model2.StateDimension());
  xdot_ans << x.tail(2*degrees_of_freedom), u;
  EXPECT_TRUE(xdot_ans.isApprox(xdot));

  // Call it as a functor
  VectorXd xdot2 = model2(x,u,1.0f);
  EXPECT_TRUE(xdot_ans.isApprox(xdot2));
}

TEST(TripleIntegrator, Jacobian) {
  int degrees_of_freedom = 2;
  TripleIntegrator model2(degrees_of_freedom);
  int n = model2.StateDimension();
  int m = model2.ControlDimension();
  double t = 0.0;
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  MatrixXd jac = MatrixXd::Zero(n,n+m);
  model2.Jacobian(x,u,t,jac);
  MatrixXd jac_ans(n,n+m);
  jac_ans << 0,0, 1,0, 0,0, 0,0,
             0,0, 0,1, 0,0, 0,0,
             0,0, 0,0, 1,0, 0,0,
             0,0, 0,0, 0,1, 0,0,
             0,0, 0,0, 0,0, 1,0,
             0,0, 0,0, 0,0, 0,1;
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
  MatrixXd hess = MatrixXd::Random(n,n);
  model2.Hessian(x,u,t,b,hess);
  EXPECT_TRUE(hess.isApproxToConstant(0.0));
}

TEST(TripleIntegrator, Discretize) {
  int degrees_of_freedom = 2;
  TripleIntegrator model_cont(degrees_of_freedom);
  DiscretizedModel<TripleIntegrator> model_discrete(model_cont);
  int n = model_discrete.StateDimension();
  int m = model_discrete.ControlDimension();
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  float t = 1.1;
  float h = 0.1;
  VectorXd xnext = model_discrete.Evaluate(x,u,t,h);
  VectorXd xdot = model_cont.Evaluate(x,u,t);
  EXPECT_GT((xdot-xnext).norm(), 1e-6);
  VectorXd k1 = model_cont.Evaluate(x, u, t) * h;
  VectorXd k2 = model_cont.Evaluate(x + k1 * 0.5, u, t) * h;
  VectorXd k3 = model_cont.Evaluate(x + k2 * 0.5, u, t) * h;
  VectorXd k4 = model_cont.Evaluate(x + k3, u, t) * h;
  VectorXd xnext2 = x + (k1 + 2*k2 + 2*k3 + k4) / 6;
  EXPECT_TRUE(xnext.isApprox(xnext2));

  VectorXd xnext3 = model_discrete(x,u,t,h);
  EXPECT_TRUE(xnext.isApprox(xnext3));

  MatrixXd jac = MatrixXd::Zero(n,n+m);
  model_discrete.Jacobian(x, u, t, h, jac);

  MatrixXd jac_cont = MatrixXd::Zero(n,n+m);
  model_cont.Jacobian(x, u, t, jac_cont);
  MatrixXd A = jac_cont.topLeftCorner(n,n);
  MatrixXd K1 = A * h; 
  MatrixXd K2 = A * h + 0.5 * A*A * pow(h, 2);
  MatrixXd K3 = A * h + 0.5 * A*A * pow(h, 2) + 0.25 * A*A*A * pow(h, 3);
  MatrixXd K4 = A * h + A*A * pow(h, 2) + 0.5 * A*A*A * pow(h, 3) + 0.25 * A*A*A*A * pow(h, 4);
  MatrixXd Ad = MatrixXd::Identity(n,n) + (K1 + 2*K2 + 2*K3 + K4) / 6;
  MatrixXd Bd(n,m);
  Bd << MatrixXd::Zero(2*m, m), MatrixXd::Identity(m,m) * h;
  EXPECT_TRUE(jac.topLeftCorner(n,n).isApprox(Ad));
  EXPECT_TRUE(jac.topRightCorner(n,m).isApprox(Bd));
}

TEST(TripleIntegrator, EulerIntegration) {
  int degrees_of_freedom = 2;
  TripleIntegrator model_cont(degrees_of_freedom);
  DiscretizedModel<TripleIntegrator,ExplicitEuler> model_discrete(model_cont);
  int n = model_discrete.StateDimension();
  int m = model_discrete.ControlDimension();
  VectorXd x = VectorXd::Random(n);
  VectorXd u = VectorXd::Random(m);
  float t = 1.1;
  float h = 0.1;
  VectorXd xnext = model_discrete.Evaluate(x,u,t,h);
  EXPECT_TRUE(xnext.isApprox(x + model_cont(x, u, t) * h));

  MatrixXd jac = MatrixXd::Zero(n,n+m);
  model_discrete.Jacobian(x, u, t, h, jac);
  MatrixXd jac_ans(n, n+m);
  jac_ans << 1,0, h,0, 0,0, 0,0,
             0,1, 0,h, 0,0, 0,0,
             0,0, 1,0, h,0, 0,0,
             0,0, 0,1, 0,h, 0,0,
             0,0, 0,0, 1,0, h,0,
             0,0, 0,0, 0,1, 0,h;
  EXPECT_TRUE(jac.isApprox(jac_ans));

}
  
} // namespace dynamics_examples
} // namespace altro
