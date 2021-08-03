add_library(constraints
  constraint.hpp
  constraint.cpp
  constraint_values.hpp
  constraint_values.cpp
)
target_link_libraries(constraints
  PRIVATE
  common
  utils
)
export_library(constraints altroTargets "constraint.hpp;constraint_values.hpp" altro/constraints)
