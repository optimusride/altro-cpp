add_library(common
  knotpoint.cpp
  knotpoint.hpp
  trajectory.cpp
  trajectory.hpp
)
target_link_libraries(common 
	PUBLIC
  Eigen3::Eigen
	PRIVATE 
	utils
)
export_library(common altroTargets "knotpoint.hpp;trajectory.hpp" altro/common)
