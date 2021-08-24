add_library(utils
  assert.cpp
  assert.hpp
  derivative_checker.cpp
  derivative_checker.hpp
  utils.hpp
  benchmarking.hpp
)
target_link_libraries(utils
  PRIVATE
  fmt::fmt
)
export_library(utils altroTargets)
