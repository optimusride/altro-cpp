//
// Created by brian on 9/6/22.
//

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/chrono.h>

#include "cartpole.hpp"
#include "altro/augmented_lagrangian/al_solver.hpp"
#include "altro/problem/discretized_model.hpp"
#include "altro/utils/formatting.hpp"
#include "examples/quadratic_cost.hpp"


int main() {
  // Create Model
  Cartpole model;
  const int n = model.StateDimension();
  const int m = model.ControlDimension();

  // Check model derivatives
  bool jac_check = model.CheckJacobian();
  bool hess_check = model.CheckHessian();
  fmt::print("Jacobian check: {}\n", jac_check);
  fmt::print("Hessian check: {}\n", hess_check);

  // Discretization
  float tf = 5.0;
  int num_segments = 100;
  double h = tf / static_cast<double>(num_segments);

  // Set up the problem
  altro::problem::Problem prob(num_segments);

  // Dynamics
  using DiscreteCartpole = altro::problem::DiscretizedModel<Cartpole>;
  std::shared_ptr<DiscreteCartpole> model_ptr = std::make_shared<DiscreteCartpole>(model);

  // Set Dynamics
  for (int k = 0; k < num_segments; ++k) {
    prob.SetDynamics(model_ptr, k);
  }

  // Objective
  using altro::examples::QuadraticCost;
  Eigen::Matrix4d Q = Eigen::Vector4d::Constant(1e-2 * h).asDiagonal();
  Eigen::Matrix<double,1,1> R = Eigen::Vector<double,1>::Constant(1e-1 * h).asDiagonal();
  Eigen::Matrix4d Qf = Eigen::Vector4d::Constant(1e2).asDiagonal();
  Eigen::Vector4d xf(0, M_PI, 0, 0);
  Eigen::Vector4d x0 = Eigen::Vector4d::Zero();
  Eigen::Vector<double,1> uref = Eigen::Vector<double,1>::Zero();

  std::shared_ptr<QuadraticCost> stage_cost;
  std::shared_ptr<QuadraticCost> term_cost;
  stage_cost = std::make_shared<QuadraticCost>(QuadraticCost::LQRCost(Q, R, xf, uref));
  term_cost = std::make_shared<QuadraticCost>(QuadraticCost::LQRCost(Qf, R, xf, uref, true));

  for (int k = 0; k < num_segments; ++k) {
    prob.SetCostFunction(stage_cost, k);
  }
  prob.SetCostFunction(term_cost, num_segments);

  // Initial State
  prob.SetInitialState(x0);

  // Initial Trajectory
  using CartpoleTrajectory =  altro::Trajectory<Cartpole::NStates, Cartpole::NControls>;
  std::shared_ptr<CartpoleTrajectory> Z = std::make_shared<CartpoleTrajectory>(n, m, num_segments);
  for (int k = 0; k < num_segments; ++k) {
    Z->Control(k) = uref;
  }
  Z->SetUniformStep(h);

  // Build Solver
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<Cartpole::NStates, Cartpole::NControls> alsolver(prob);
  fmt::print("Solver initialized!\n");
  alsolver.SetTrajectory(Z);

  // Set Options
  alsolver.GetOptions().constraint_tolerance = 1e-6;
  alsolver.GetOptions().verbose = altro::LogLevel::kSilent;
  alsolver.GetOptions().max_iterations_total = 200;
  alsolver.GetiLQRSolver().GetOptions().verbose = altro::LogLevel::kInner;

  // Solve
  using millisf = std::chrono::duration<double, std::milli>;
  auto start = std::chrono::high_resolution_clock::now();
  alsolver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<millisf>(stop - start);
  fmt::print("Total time: {}\n", duration);

  fmt::print("x0: \n{}\n", Z->State(0));

  return EXIT_SUCCESS;
}