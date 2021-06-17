add_library(ilqr
	ilqr.cpp
	cost_expansion.cpp
)
target_link_libraries(ilqr
	common
	problem
	utils
	Eigen3::Eigen
)

add_subdirectory(test)