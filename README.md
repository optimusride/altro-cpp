
# AltroCpp
A non-linear trajectory optimization library developed by [Optimus Ride, Inc.](https://www.optimusride.com/)
This library implements a C++ version of the original [open-source ALTRO solver](https://github.com/RoboticExplorationLab/Altro.jl) developed by the [Robotic Exploration Lab](https://roboticexplorationlab.org/) at Stanford and Carnegie Mellon Universities, also available open-source as an official Julia package.

For details on the algorithm, see the related papers and tutorial: 
- [Tutorial](https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf)
- [Original Paper](https://roboticexplorationlab.org/papers/altro-iros.pdf)
- [Conic MPC Paper](https://roboticexplorationlab.org/papers/ALTRO_MPC.pdf)
- [Planning with Attitude](https://roboticexplorationlab.org/papers/planning_with_attitude.pdf)


## License 

```
Copyright [2021] Optimus Ride Inc.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
```

## Building from Source
### Install Build Dependencies
The build depends on cmake, Eigen, fmt and doxygen. On Debian based systems, use the following command to install build dependencies:
```bash
sudo apt-get install cmake libeigen3-dev libfmt-dev doxygen
```

### Build
This library uses the CMake build system. To build the source code and compile the library,
follow the canonical CMake usage.
```bash
cd altro-cpp         # Change directory into directory root.
mkdir build          # Create a build directory.
cmake ..             # Run the CMake configuration step. 
cmake --build .      # Build all CMake targets
```

To run with a different generator (such as Ninja), you can specify the generator when running the configuration step:
```bash
cmake -G Ninja ..
```

### Build Options
The build system provides the following build options:
- `ALTRO_RUN_CLANG_TIDY` - Run static analysis on source code. Must be built with the `clang` compiler. (Default = `OFF`)
- `ALTRO_BUILD_TESTS` - Build the test suite. (Default = `ON`)
- `ALTRO_BUILD_EXAMPLES` - Build the code in the `examples` directory. (Default = `ON`)
- `ALTRO_BUILD_BENCHMARKS` - Build the benchmark problems located in the `perf` directory. (Default = `ON`)
- `ALTRO_BUILD_COVERAGE` - Run a code coverage analysis on the test suite (experimental). (Default = `OFF`)
- `ALTRO_SET_POSITION_INDEPENDENT` - Set the `-fPIC` option to compile the code as position independent. Often needed when incorporating into other libraries. (Default = `ON`)
- `ALTRO_BUILD_SHARED_LIBS` - Build all the libraries as shared libraries instead of static libraries. (Default = `OFF`)

Any of these options can be specified at configuration time by passing them wth the `-D` flag:
```bash
cmake -D OPTION1=OFF -D OPTION2=ON ..
```

Alternatively, these can be modified using `ccmake` or `cmake-gui`, e.g.
```bash
cmake-gui ..
```
or, if the configuration step has already been run, you can always run the configuration step on the build directory, as well:
```bash
cmake-gui .
```

### Installation
The build system provides an option to install the compiled libraries and header files onto your local system. The install location is controlled via the `CMAKE_INSTALL_PREFIX` cache variable,
which defaults to `~/.local`.

To install and (optionally) specify the install location, build the
`install` target:
```bash
cmake -DCMAKE_INSTALL_PREFIX=~/.local  # or use cmake-gui
cmake --build . --target install       # build the install target
```
The install will create the following structure below `CMAKE_INSTALL_PREFIX`:
```
include/
  altro/
    augmented_lagrangian/
    common/
    ...
lib/
  cmake/
    AltroCpp/
      AltroCppConfig.cmake
      AltroCppConfigVersion.cmake
      AltroCppTargets.cmake
      ...
  libaugmented_lagrangian.a  # Assuming static libraries
  libcommon.a
  ...
```

### Running Unit Tests
The unit test suite can easily be run using CTest. In the `build/` directory, run the following command:
```bash
ctest .
```

### Building the Documentation locally
The documentation can be built using the `doxygen` target:
```bash
cmake --build . --target doxygen
```
The home page is located at 
```bash
build/docs/html/index.html
```

## Using the library
The easiest way to use the library is to pull the CMake targets into an exsiting CMake build system. If the library is installed locally, this is done via `find_package`:
```cmake
set(AltroCpp_DIR ~/.local)  # or wherever the user installed the library via CMAKE_INSTALL_PREFIX
find_library(AltroCpp 0.3 REQUIRED EXACT)
```

The `REQUIRED` and `EXACT` arguments can be left off, if needed. See
CMake documentation on `find_package` for more details.

Once the library is found, the user can link against the library using the `altro::altro` target:

```cmake
add_executable(my_program
  main.cpp
)
target_link_libraries(my_program
  PRIVATE
  altro::altro
)
```

The library automatically adds the `include` install directory to the include path, so all `#include` statements should be relative to 
the altro root directory, e.g.
```cpp
#include <iostream>
#include <altro/augmented_lagrangian/al_solver.hpp>

int main() {
  const int NumStates = 4;
  const int NumControls = 2;
  int num_segments = 100;
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<NumStates, NumControls> solver(num_segments);

  return 0;
}
```
