// Copyright [2021] Optimus Ride Inc.

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
#include "examples/problems/unicycle.hpp"

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

  for (int k = 0; k < N; ++k) {
    prob.SetDynamics(MakeModel(dof), k);
    prob.SetCostFunction(MakeCost(dof), k);
  }
  prob.SetCostFunction(MakeCost(dof), N);
  prob.SetInitialState(x0);

  return prob;
}

}