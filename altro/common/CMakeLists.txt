add_library(common
  knotpoint.cpp
  knotpoint.hpp
  state_control_sized.hpp
  trajectory.cpp
  trajectory.hpp
)
target_link_libraries(common 
	PUBLIC
  Eigen3::Eigen
  INTERFACE
  fmt::fmt-header-only
	PRIVATE 
	utils
)
export_library(common altroTargets "knotpoint.hpp;trajectory.hpp" altro/common)
