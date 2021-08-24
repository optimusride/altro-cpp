# Prepares a target for installation and exporting
# This function does the following:
# 1. Adds the current directory to the include paths for the target at build
#    time
# 2. Adds the top-level install include directory to the include path for the 
#    installed target
# 3. Adds the target to the `export_name` export set
# 4. Installs the target at the default locations.
# 
# ARGUMENTS
# - lib_name - name of the target (typically a library).
# - export_name - name of the export set the target should be added to
function(export_library lib_name export_name)
  # This means that any consumer of the exported target will have the 
  # installed include directory automatically added to their include path.
  target_include_directories(${lib_name}
                           INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  )

  # Install the target and add to the export set
  install(TARGETS ${lib_name}
        EXPORT ${export_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endfunction()