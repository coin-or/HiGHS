#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenBLAS::OpenBLAS" for configuration "RelWithDebInfo"
set_property(TARGET OpenBLAS::OpenBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/openblas.lib"
  )

list(APPEND _cmake_import_check_targets OpenBLAS::OpenBLAS )
list(APPEND _cmake_import_check_files_for_OpenBLAS::OpenBLAS "${_IMPORT_PREFIX}/lib/openblas.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
