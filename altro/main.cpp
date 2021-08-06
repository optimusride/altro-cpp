#include <iostream>
#include <vector>
#include "altro/common/trajectory.hpp"

int main() {
  float a = 10.0f;
  std::vector<float> vec = {1, 4, a};
  std::vector<float> vec2 = vec;
  std::cout << "Hello, from altro!\n";
  return EXIT_SUCCESS;
}
