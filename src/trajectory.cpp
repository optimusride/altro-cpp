#include "trajectory.hpp"

namespace altro {

void PrintMatrix()
{
  Eigen::MatrixXd m(2,2);
  m << 1, 2, 3, 4;
  std::cout << m << std::endl;
}

} // namespace altro