
# Altro

  A non-linear trajectory optimization library. For more details: https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf

## Installation
  Run the following script: 
    
    ./install.sh     
  This configures the makefile, builds the code and installs it in a location specified in the cmake file. 

  To use it in vehicle repo, please make the following changes:	

   In cmake file:
         
         find_package(altro REQUIRED)

  To link against altro:

        ori_cc_library(motion_status
        SOURCES
          motion_status.cpp
        LIBS
          altro::altro
        )

  In the source file include the file name directly:
	
     #include "dynamics.hpp"


## Running unit tests
All the unit tests can be run by running 

    ./run_tests.sh
  

## Building documentation
To build the documentation, run the following commands from the root directory:
```
mkdir build
cd build
cmake ..
cmake --build . --target doxygen
```
The documentation will be built locally in `build/html`. Open the `build/html/index.html` 
file in your browser to view the documentation.