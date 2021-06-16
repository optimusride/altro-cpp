#pragma once

#include <eigen3/Eigen/Dense>

namespace altro {

template <int n, class T = double>
using Vector = Eigen::Matrix<T, n, 1>;
using VectorXd = Eigen::VectorXd;
using VectorXf = Eigen::VectorXf;

using MatrixXd = Eigen::MatrixXd;
using MatrixXf = Eigen::MatrixXf;

}  // namespace altro 