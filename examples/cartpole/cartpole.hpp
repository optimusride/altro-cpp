//
// Created by brian on 9/6/22.
//

#ifndef ALTROCPP_EXAMPLES_CARTPOLE_CARTPOLE_HPP_
#define ALTROCPP_EXAMPLES_CARTPOLE_CARTPOLE_HPP_

#include "altro/problem/dynamics.hpp"

class Cartpole : public altro::problem::ContinuousDynamics {
 public:
  Cartpole() = default;
  static constexpr int NStates = 4;
  static constexpr int NControls = 1;

  int StateDimension() const override { return NStates; }
  int ControlDimension() const override { return NControls; }

  void Evaluate(const altro::VectorXdRef& x, const altro::VectorXdRef& u, const float t,
                Eigen::Ref<Eigen::VectorXd> xdot) override;

  void Jacobian(const altro::VectorXdRef& x, const altro::VectorXdRef& u, const float t,
                Eigen::Ref<Eigen::MatrixXd> jac) override;

  void Hessian(const altro::VectorXdRef& x, const altro::VectorXdRef& u, const float t,
               const altro::VectorXdRef& b,
               Eigen::Ref<Eigen::MatrixXd> hess) override;

  bool HasHessian() const override { return true; }

 private:
  double mass_cart_ = 1.0;
  double mass_pole_ = 0.2;
  double length_ = 0.5;
  double gravity_ = 9.81;
};


#endif  // ALTROCPP_EXAMPLES_CARTPOLE_CARTPOLE_HPP_
