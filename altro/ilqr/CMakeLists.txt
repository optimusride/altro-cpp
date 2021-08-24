add_library(ilqr
  cost_expansion.cpp
  cost_expansion.hpp
  dynamics_expansion.cpp
  dynamics_expansion.hpp
  knot_point_function_type.cpp
  knot_point_function_type.hpp
  ilqr.cpp
  ilqr.hpp
)
target_link_libraries(ilqr
  PUBLIC
  fmt::fmt
  Eigen3::Eigen
  threadpool

  PRIVATE
  common
  problem
  utils
)
export_library(ilqr altroTargets)