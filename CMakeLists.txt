cmake_minimum_required(VERSION 3.12.1)
project(altro VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -pedantic -Werror)

# Defines GNU standard installation directories
include(GNUInstallDirs)
include(CMakePrintHelpers)
include(FindThreads)

# Check if MacOS
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)

    # Add the Homebrew headers for Big Sur
    include_directories( "/opt/homebrew/include/" )
endif()

################################################################################
############################### OPTIONS ########################################
################################################################################
option(RUN_CLANG_TIDY "Enable static analysis w/ clang tidy" OFF)
option(BUILD_TESTS "Build the tests" ON)
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_COVERAGE "Build test coverage" OFF)
option(BUILD_BENCHMARKS "Build the benchmark problems" ON)

# Set (and create) log file directory
set(LOGDIR "${PROJECT_BINARY_DIR}/logs" CACHE FILESTRING "Location of log files")
file(MAKE_DIRECTORY ${LOGDIR})

# Examples are required to build the tests
if (BUILD_TESTS)
  set(BUILD_EXAMPLES ON)
endif()


################################################################################
############################ ADD EIGEN LIB #####################################
################################################################################
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

################################################################################
############################ ADD FMT LIBRARY ###################################
################################################################################
# NOTE(bjackson): Can we use a newer version? Should we bundle it with the 
# source code?
set(fmt_ROOT "/opt/fmt")
set(fmt_REQUIRED_VERSION 6.2.1)
set(fmt_REQUIRED_DEBIAN_VERSION ${fmt_REQUIRED_VERSION})
find_package(fmt ${fmt_REQUIRED_VERSION} REQUIRED)

################################################################################
########################## ENABLE GOOGLE TEST ##################################
################################################################################

# This flag is set to ensure that gtest and gmock files do not get installed along with rest of the package. 
set(INSTALL_GTEST off CACHE BOOL "" FORCE)

# Add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()
enable_testing()
include(GoogleTest)

################################################################################
############################ DOCUMENTATION #####################################
################################################################################

include(FindDoxygen)
set(DOXYGEN_USE_MATHJAX YES)
doxygen_add_docs(doxygen 
  ${PROJECT_SOURCE_DIR}/altro
)

################################################################################
############################ CODE COVERAGE #####################################
################################################################################
# Code Coverage
# Build by:
# cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON ..
# Run: "make coverage" and open the build/altro_test_coverage.html
# Currently all .html and .css are littered in /build
# TODO(jw) : move all report files to a dedicated sub-directory


if(BUILD_COVERAGE)
    if (MACOSX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    endif (MACOSX)

add_custom_target(coverage ALL
    COMMAND gcovr -r ../ --html-details --exclude-directories='_deps' -o altro_test_coverage.html
)

endif(BUILD_COVERAGE)


################################################################################
############################# BUILD STEPS ######################################
################################################################################

# Sources
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR})


# Macro to allow exporting a library
macro(export_library lib_name export_name headers relative_path)
  set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER "${headers}")
  target_include_directories(${lib_name}
                           INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )
  install(TARGETS ${lib_name}
        EXPORT ${export_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${relative_path}
  )
endmacro()

# This adds the -fPIC option for all build targets. The position independent code is required for the library to be usable with vehicle repo. 
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(altro)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if (BUILD_TESTS)
  add_subdirectory(test)
endif()
if (BUILD_BENCHMARKS)
  add_subdirectory(perf)
endif()


################################################################################
############################# INSTALLATION #####################################
################################################################################
# NOTE: The following steps allow installing this project for external use. These steps should be done from the outer-most cmake file only.

# Ensure that the `CMAKE_INSTALL_PREFIX` is set correctly. This is the place where the package headers and libraries will get installed. This same prefix is also used by `find_package` cmake module. 
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "~/.local" CACHE PATH "..." FORCE)
endif()

# Export the altroTargets.cmake file
install(EXPORT altroTargets
  FILE altroTargets.cmake
  NAMESPACE altro::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/altro
)

# Generate the package config file that includes the exported targets
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/altroConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/altro
)

# Generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/altroConfigVersion.cmake"
  VERSION "${altro_VERSION_MAJOR}.${altro_VERSION_MINOR}.${altro_VERSION_PATCH}"
  COMPATIBILITY AnyNewerVersion
)

# Install the configuration file and the version file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/altro
)

install(EXPORT altroTargets
       FILE altroTargets.cmake
       NAMESPACE altro::
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
