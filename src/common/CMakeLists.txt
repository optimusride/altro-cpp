add_library(common
  trajectory.cpp
  knotpoint.cpp
)
target_link_libraries(common 
	PUBLIC
  Eigen3::Eigen

	PRIVATE 
	utils
)
