#pragma once

#include <eigen3/Eigen/Dense>

namespace altro {

template <int n, class T = double>
using VectorN = Eigen::Matrix<T, n, 1>;

template <int n>
using VectorNd = Eigen::Matrix<double, n, 1>;

template <int n, int m>
using MatrixNxMd = Eigen::Matrix<double, n, m>;

typedef Eigen::Ref<const Eigen::VectorXd> VectorXdRef; 

using VectorXd = Eigen::VectorXd;
using VectorXf = Eigen::VectorXf;

using MatrixXd = Eigen::MatrixXd;
using MatrixXf = Eigen::MatrixXf;

}  // namespace altro 