# Install script for directory: C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/galab/installs/highs")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/interface/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/driver/level2/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/driver/level3/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/driver/others/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/kernel/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/lapack/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/utest/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/ctest/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/lib/DEBUG/openblas.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/lib/RELEASE/openblas.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/lib/MINSIZEREL/openblas.lib")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/lib/RELWITHDEBINFO/openblas.lib")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openblas" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/openblas_config.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openblas" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/generated/cblas.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openblas" TYPE FILE FILES
    "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src/lapack-netlib/LAPACKE/example/lapacke_example_aux.h"
    "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src/lapack-netlib/LAPACKE/include/lapack.h"
    "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src/lapack-netlib/LAPACKE/include/lapacke.h"
    "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src/lapack-netlib/LAPACKE/include/lapacke_config.h"
    "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src/lapack-netlib/LAPACKE/include/lapacke_mangling.h"
    "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-src/lapack-netlib/LAPACKE/include/lapacke_utils.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openblas" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/lapacke_mangling.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/openblas.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/OpenBLASConfig.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE RENAME "OpenBLASConfigVersion.cmake" FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/OpenBLASConfigVersion.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS/OpenBLASTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS/OpenBLASTargets.cmake"
         "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/CMakeFiles/Export/2d44cc56bb35d712767263736bedc37c/OpenBLASTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS/OpenBLASTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS/OpenBLASTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/CMakeFiles/Export/2d44cc56bb35d712767263736bedc37c/OpenBLASTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/CMakeFiles/Export/2d44cc56bb35d712767263736bedc37c/OpenBLASTargets-debug.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/CMakeFiles/Export/2d44cc56bb35d712767263736bedc37c/OpenBLASTargets-minsizerel.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/CMakeFiles/Export/2d44cc56bb35d712767263736bedc37c/OpenBLASTargets-relwithdebinfo.cmake")
  endif()
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/OpenBLAS" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/CMakeFiles/Export/2d44cc56bb35d712767263736bedc37c/OpenBLASTargets-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/galab/code/HiGHS/build-32/_deps/openblas-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
