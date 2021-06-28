#pragma once 

#include <memory>

#include "altro/problem/discretized_model.hpp"
#include "altro/problem/problem.hpp"
#include "examples/triple_integrator.hpp"
#include "examples/quadratic_cost.hpp"

namespace altro{

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

}