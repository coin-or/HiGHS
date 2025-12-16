#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "highs::highs" for configuration "MinSizeRel"
set_property(TARGET highs::highs APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(highs::highs PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "C;CXX;RC"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/highs.lib"
  )

list(APPEND _cmake_import_check_targets highs::highs )
list(APPEND _cmake_import_check_files_for_highs::highs "${_IMPORT_PREFIX}/lib/highs.lib" )

# Import target "highs::OpenBLAS" for configuration "MinSizeRel"
set_property(TARGET highs::OpenBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(highs::OpenBLAS PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "C"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/openblas.lib"
  )

list(APPEND _cmake_import_check_targets highs::OpenBLAS )
list(APPEND _cmake_import_check_files_for_highs::OpenBLAS "${_IMPORT_PREFIX}/lib/openblas.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
