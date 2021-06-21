#!/bin/bash

# TODO(shivang) : This deletes all the files from installed location but still keeps the directory
# structure. See this for more details:
# https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#can-i-do-make-uninstall-with-cmake
xargs rm < build/install_manifest.txt
