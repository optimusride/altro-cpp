add_library(utils
  assert.cpp
  assert.hpp
  derivative_checker.cpp
  derivative_checker.hpp
  utils.hpp
  benchmarking.hpp
)
export_library(utils altroTargets "assert.hpp;derivative_checker.hpp;utils.hpp" altro/utils)
