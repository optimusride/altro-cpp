#pragma once 

#include <memory>

#include "altro/augmented_lagrangian/al_problem.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/constraints/constraint.hpp"
#include "altro/ilqr/cost_expansion.hpp"
#include "altro/ilqr/ilqr.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/problem/problem.hpp"
#include "examples/basic_constraints.hpp"
#include "examples/quadratic_cost.hpp"
#include "examples/unicycle.hpp"
#include "examples/triple_integrator.hpp"

namespace altro {

using ModelType = problem::DiscretizedModel<examples::TripleIntegrator>;
using ModelPtr = std::shared_ptr<ModelType>;
using CostPtr = std::shared_ptr<examples::QuadraticCost>;

ModelPtr MakeModel(int dof = 2) {
  examples::TripleIntegrator model(dof);
  return std::make_shared<ModelType>(model);
}

CostPtr MakeCost(int dof = 2) {
  MatrixXd Q = VectorXd::LinSpaced(3 * dof, 1, 3 * dof - 1).asDiagonal();
  MatrixXd R = VectorXd::LinSpaced(dof, 1, dof - 1).asDiagonal();
  MatrixXd H = MatrixXd::Zero(3 * dof, dof);
  VectorXd q = VectorXd::LinSpaced(3 * dof, 1, 3 * dof - 1);
  VectorXd r = VectorXd::LinSpaced(dof, 1, dof - 1);
  double c = 11;
  return std::make_shared<examples::QuadraticCost>(Q, R, H, q, r, c);
}

problem::Problem MakeProblem(int dof = 2, int N = 10) {
  problem::Problem prob(N);
  CostPtr costfun_ptr = MakeCost(dof);
  ModelPtr model_ptr = MakeModel(dof); 
  VectorXd x0 = VectorXd::Random(3 * dof);

  prob.SetDynamics(model_ptr, 0, N);
  prob.SetCostFunction(costfun_ptr, 0, N+1);
  prob.SetInitialState(x0);

  return prob;
}

class UnicycleProblem {
 public:
  UnicycleProblem() {

    // Dynamics
    model = std::make_shared<problem::DiscretizedModel<examples::Unicycle>>(examples::Unicycle());

    // Cost functions
    Q *= h;  // scale by time step to match Altro.jl
    R *= h;
    qcost = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Q, R, xf, uref));
    qterm = std::make_shared<examples::QuadraticCost>(
        examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));

    // Constraints
    goal = std::make_shared<examples::GoalConstraint>(xf);
    std::vector<double> lb = {-v_bnd, -w_bnd};
    std::vector<double> ub = {v_bnd, w_bnd};
    ubnd = std::make_shared<examples::ControlBound>(lb, ub);
  }
  virtual ~UnicycleProblem() = default;

 protected:
  using ModelType = altro::problem::DiscretizedModel<altro::examples::Unicycle>;
  using CostFunType = altro::examples::QuadraticCost;

  static constexpr int n_static = 3;
  static constexpr int m_static = 2;
  static constexpr int HEAP = Eigen::Dynamic;

  int N = 100;
  int n = n_static;
  int m = m_static;
  float tf = 3.0;
  float h = 0.03;
  std::shared_ptr<problem::DiscretizedModel<examples::Unicycle>> model;

  Eigen::Matrix3d Q = Eigen::Vector3d::Constant(n_static, 1e-2).asDiagonal();
  Eigen::Matrix2d R = Eigen::Vector2d::Constant(m_static, 1e-2).asDiagonal();
  Eigen::Matrix3d Qf = Eigen::Vector3d::Constant(n_static, 100).asDiagonal();
  Eigen::Vector3d xf = Eigen::Vector3d(1.5, 1.5, M_PI / 2);
  Eigen::Vector3d x0 = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector2d u0 = Eigen::Vector2d::Constant(m_static, 0.1);
  Eigen::Vector2d uref = Eigen::Vector2d::Zero();
  std::shared_ptr<examples::QuadraticCost> qcost;
  std::shared_ptr<examples::QuadraticCost> qterm;

  double v_bnd = 1.5;
  double w_bnd = 1.5;
  altro::constraints::ConstraintPtr<altro::constraints::Inequality> ubnd;
  altro::constraints::ConstraintPtr<altro::constraints::Equality> goal;


  altro::problem::Problem MakeProblem() {
    altro::problem::Problem prob(N);

    // Cost Function
    prob.SetCostFunction(qcost, 0, N);
    prob.SetCostFunction(qterm, N);

    // Dynamics
    prob.SetDynamics(model, 0, N);

    // Constraints
    std::vector<double> lb = {-v_bnd, -w_bnd};
    std::vector<double> ub = {+v_bnd, +w_bnd};
    ubnd = std::make_shared<altro::examples::ControlBound>(lb, ub);
    for (int k = 0; k < N; ++k) {
      prob.SetConstraint(ubnd, k);
    }
    goal = std::make_shared<altro::examples::GoalConstraint>(xf);
    prob.SetConstraint(goal, N);

    // Initial State
    prob.SetInitialState(x0);

    return prob;
  }

  template <int n_size, int m_size>
  altro::Trajectory<n_size, m_size> InitialTrajectory() {
    altro::Trajectory<n_size, m_size> Z(n, m, N);
    for (int k = 0; k < N; ++k) {
      Z.Control(k) = u0;
    }
    Z.SetUniformStep(h);
    return Z;
  }

  template <int n_size, int m_size>
  altro::ilqr::iLQR<n_size, m_size> MakeSolver(const bool alcost = false) {
    altro::problem::Problem prob = MakeProblem();
    if (alcost) {
      prob = altro::augmented_lagrangian::BuildAugLagProblem<n_size, m_size>(prob);
    }
    altro::ilqr::iLQR<n_size, m_size> solver(prob);

    std::shared_ptr<altro::Trajectory<n_size, m_size>> traj_ptr =
        std::make_shared<altro::Trajectory<n_size, m_size>>(InitialTrajectory<n_size, m_size>());

    solver.SetTrajectory(traj_ptr);
    return solver;
  }
};

}