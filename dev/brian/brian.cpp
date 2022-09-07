//
// Created by Brian Jackson on 9/7/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//
#include <stdlib.h>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <Eigen/Dense>

#include "altro/utils/formatting.hpp"

int main() {
  Eigen::VectorXd x(3);
  x << 1, 2, 3;
  Eigen::IOFormat format(Eigen::StreamPrecision, 0, " ", "\n", "[", "]", "", "");
  fmt::print("Hi there Brian!\n");
  fmt::print("Print x:\n{}\n", x.format(format));
  return EXIT_SUCCESS;
}
