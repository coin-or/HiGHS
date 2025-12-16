#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "highs::highs" for configuration "Debug"
set_property(TARGET highs::highs APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(highs::highs PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C;CXX;RC"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/highs.lib"
  )

list(APPEND _cmake_import_check_targets highs::highs )
list(APPEND _cmake_import_check_files_for_highs::highs "${_IMPORT_PREFIX}/lib/highs.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
