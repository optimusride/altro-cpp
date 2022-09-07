add_library(utils
  assert.cpp
  assert.hpp
  derivative_checker.cpp
  derivative_checker.hpp
  utils.hpp
  benchmarking.hpp
  formatting.cpp
  formatting.hpp
)
target_link_libraries(utils
  PRIVATE
  fmt::fmt
  Eigen3::Eigen
)
export_library(utils AltroCppTargets)
