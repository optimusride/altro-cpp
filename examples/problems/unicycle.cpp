#include "examples/problems/unicycle.hpp"

namespace altro {
namespace problems {

UnicycleProblem::UnicycleProblem() {
  // Dynamics
  model = std::make_shared<problem::DiscretizedModel<examples::Unicycle>>(examples::Unicycle());

  // Cost functions
  Q *= h;  // scale by time step to match Altro.jl
  R *= h;
}

altro::problem::Problem UnicycleProblem::MakeProblem(const bool add_constraints) {
  altro::problem::Problem prob(N);

  goal = std::make_shared<altro::examples::GoalConstraint>(xf);

  if (scenario_ == kTurn90) {
    tf = 3.0;
    h = tf / N;

    std::vector<double> lb = {-v_bnd, -w_bnd};
    std::vector<double> ub = {+v_bnd, +w_bnd};
    ubnd = std::make_shared<altro::examples::ControlBound>(lb, ub);

  } else if (scenario_ == kThreeObstacles) {
    tf = 5.0;
    h = tf / N;

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

    std::shared_ptr<altro::examples::CircleConstraint> obs =
        std::make_shared<altro::examples::CircleConstraint>();
    for (int i = 0; i < num_obstacles; ++i) {
      obs->AddObstacle(cx(i), cy(i), cr(i));
    }
    obstacles = std::move(obs);

    std::vector<double> lb = {0, -3};
    std::vector<double> ub = {3, +3};

    ubnd = std::make_shared<altro::examples::ControlBound>(lb, ub);
    for (int k = 1; k < N; ++k) {
      prob.SetConstraint(obstacles, k);
    }
  }

  // Cost Function
  qcost =
      std::make_shared<examples::QuadraticCost>(examples::QuadraticCost::LQRCost(Q, R, xf, uref));
  qterm = std::make_shared<examples::QuadraticCost>(
      examples::QuadraticCost::LQRCost(Qf, R * 0, xf, uref, true));
  for (int k = 0; k < N; ++k) {
    prob.SetCostFunction(qcost, k);
  }
  prob.SetCostFunction(qterm, N);

  // Dynamics
  for (int k = 0; k < N; ++k) {
    prob.SetDynamics(model, k);
  }

  // Constraints
  if (add_constraints) {
    goal = std::make_shared<examples::GoalConstraint>(xf);
    for (int k = 0; k < N; ++k) {
      prob.SetConstraint(ubnd, k);
    }
    prob.SetConstraint(goal, N);
  }

  // Initial State
  prob.SetInitialState(x0);

  return prob;
}

}  // namespace problems
}  // namespace altro