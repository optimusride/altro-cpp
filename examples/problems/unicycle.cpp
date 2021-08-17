#include "examples/problems/unicycle.hpp"

namespace altro {
namespace problems {

UnicycleProblem::UnicycleProblem() {
}

altro::problem::Problem UnicycleProblem::MakeProblem(const bool add_constraints) {
  altro::problem::Problem prob(N);

  // goal = std::make_shared<altro::examples::GoalConstraint>(xf);

  float h;
  if (scenario_ == kTurn90) {
    tf = 3.0;
    h = GetTimeStep();

    lb = {-v_bnd, -w_bnd};
    ub = {+v_bnd, +w_bnd};
    Q.diagonal().setConstant(1e-2 * h);
    R.diagonal().setConstant(1e-2 * h);
    Qf.diagonal().setConstant(100.0);

  } else if (scenario_ == kThreeObstacles) {
    tf = 5.0;
    h = GetTimeStep();

    Q.diagonal().setConstant(1.0 * h);
    R.diagonal().setConstant(0.5 * h);
    Qf.diagonal().setConstant(10.0);
    x0.setZero();
    xf << 3, 3, 0;
    u0.setConstant(0.01);

    const double scaling = 3.0;
    constexpr int num_obstacles = 3;
    cx = Eigen::Vector3d(0.25, 0.5, 0.75);  // x-coordinates of obstacles
    cy = Eigen::Vector3d(0.25, 0.5, 0.75);  // y-coordinates of obstacles
    cr = Eigen::Vector3d::Constant(0.425);  // radii of obstacles
    cx *= scaling;
    cy *= scaling;

    altro::examples::CircleConstraint obs;
    for (int i = 0; i < num_obstacles; ++i) {
      obs.AddObstacle(cx(i), cy(i), cr(i));
    }
    obstacles = std::move(obs);

    lb = {0, -3};
    ub = {3, +3};

    for (int k = 1; k < N; ++k) {
      std::shared_ptr<altro::constraints::Constraint<altro::constraints::Inequality>> obs =
          std::make_shared<altro::examples::CircleConstraint>(obstacles);
      prob.SetConstraint(obs, k);
    }
  }

  // Cost Function
  for (int k = 0; k < N; ++k) {
    qcost =
        std::make_shared<examples::QuadraticCost>(examples::QuadraticCost::LQRCost(Q, R, xf, uref));
    prob.SetCostFunction(qcost, k);
  }
  qterm = std::make_shared<examples::QuadraticCost>(
      examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
  prob.SetCostFunction(qterm, N);

  // Dynamics
  for (int k = 0; k < N; ++k) {
    prob.SetDynamics(std::make_shared<ModelType>(model), k);
  }

  // Constraints
  if (add_constraints) {
    for (int k = 0; k < N; ++k) {
      prob.SetConstraint(std::make_shared<altro::examples::ControlBound>(lb, ub), k);
    }
    prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), N);
  }

  // Initial State
  prob.SetInitialState(x0);

  return prob;
}

}  // namespace problems
}  // namespace altro