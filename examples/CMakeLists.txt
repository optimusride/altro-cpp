include_directories(${PROJECT_SOURCE_DIR}/examples)

add_library(costfunctions
  quadraticcost.cpp
)
target_link_libraries(costfunctions
  altro
  utils
)

add_subdirectory(test)