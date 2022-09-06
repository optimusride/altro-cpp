//
// Created by brian on 9/6/22.
//

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "cartpole.hpp"
#include "altro/augmented_lagrangian/al_solver.hpp"


int main() {
  Cartpole model;

  bool jac_check = model.CheckJacobian();
  bool hess_check = model.CheckHessian();
  fmt::print("Jacobian check: {}\n", jac_check);
  fmt::print("Hessian check: {}\n", hess_check);
  return EXIT_SUCCESS;
}