# Define library.

if (BUILD_CXX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_PROJECT_DIR}/cmake")
  include(source)
  set(sources ${highs_sources} ${cupdlp_sources} ${ipx_sources} ${basiclu_sources})
  set(headers ${highs_headers} ${cupdlp_headers} ${ipx_headers} ${basiclu_headers})

  # Outdated CMake approach: update in progress
  if(NOT FAST_BUILD)
    include_directories(ipm/ipx)
    include_directories(ipm/basiclu)
    include_directories(pdlp/cupdlp)

    add_library(libhighs ${sources} ${headers})

    if(${BUILD_SHARED_LIBS})
      # put version information into shared library file
      set_target_properties(libhighs PROPERTIES
        VERSION
        ${HIGHS_VERSION_MAJOR}.${HIGHS_VERSION_MINOR}.${HIGHS_VERSION_PATCH}
        SOVERSION ${HIGHS_VERSION_MAJOR}.${HIGHS_VERSION_MINOR})
      if(MINGW)
        target_compile_definitions(libhighs PUBLIC LIBHIGHS_STATIC_DEFINE)
      endif()
    else()
      # create static highs library with pic
      set_target_properties(libhighs PROPERTIES
        POSITION_INDEPENDENT_CODE on)
      target_compile_definitions(libhighs PUBLIC LIBHIGHS_STATIC_DEFINE)
    endif()

    # on UNIX system the 'lib' prefix is automatically added
    set_target_properties(libhighs PROPERTIES
      OUTPUT_NAME "highs"
      MACOSX_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

    if(ZLIB AND ZLIB_FOUND)
      target_link_libraries(libhighs ZLIB::ZLIB)
      set(CONF_DEPENDENCIES "include(CMakeFindDependencyMacro)\nfind_dependency(ZLIB)")
    endif()

    # set the install rpath to the installed destination
    set_target_properties(libhighs PROPERTIES INSTALL_RPATH
      "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

    # install the header files of highs
    foreach(file ${headers})
      get_filename_component(dir ${file} DIRECTORY)

      if(NOT dir STREQUAL "")
        string(REPLACE ../extern/ "" dir ${dir})
      endif()

      install(FILES ${file} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs/${dir})
    endforeach()
    install(FILES ${HIGHS_BINARY_DIR}/HConfig.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs)

    if(UNIX)
      # target_compile_options(libhighs PRIVATE "-Wno-defaulted-function-deleted")
      # target_compile_options(libhighs PRIVATE "-Wno-return-type-c-linkage")
      target_compile_options(libhighs PRIVATE "-Wno-return-type" "-Wno-switch")

      target_compile_options(libhighs PRIVATE "-Wno-unused-variable")
      target_compile_options(libhighs PRIVATE "-Wno-unused-const-variable")

      # target_compile_options(libhighs PRIVATE "-Wno-sign-compare")
      # target_compile_options(libhighs PRIVATE "-Wno-logical-op-parentheses")

      # target_compile_options(libipx PRIVATE "-Wno-defaulted-function-deleted")
      # target_compile_options(libipx PRIVATE "-Wno-return-type-c-linkage")
      # target_compile_options(libipx PRIVATE "-Wno-return-type" "-Wno-switch")

      # target_compile_options(libipx PRIVATE "-Wno-sign-compare")
      # target_compile_options(libipx PRIVATE "-Wno-logical-op-parentheses")
    endif()
    
    install(TARGETS libhighs EXPORT highs-targets
      LIBRARY
      ARCHIVE
      RUNTIME
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs)

    # Add library targets to the build-tree export set
    export(TARGETS libhighs
      FILE "${HIGHS_BINARY_DIR}/highs-targets.cmake")

    # Configure the config file for the build tree:
    # Either list all the src/* directories here, or put explicit paths in all the
    # include statements.
    # M reckons that the latter is more transparent, and I'm inclined to agree.
    set(CONF_INCLUDE_DIRS "${HIGHS_SOURCE_DIR}/src" "${HIGHS_BINARY_DIR}")
    configure_file(${HIGHS_SOURCE_DIR}/highs-config.cmake.in
      "${HIGHS_BINARY_DIR}/highs-config.cmake" @ONLY)

    # Configure the config file for the install
    set(CONF_INCLUDE_DIRS "\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_INCLUDEDIR}/highs")
    configure_file(${HIGHS_SOURCE_DIR}/highs-config.cmake.in
      "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs-config.cmake" @ONLY)

    # Configure the pkg-config file for the install
    configure_file(${HIGHS_SOURCE_DIR}/highs.pc.in
      "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs.pc" @ONLY)

    # Install the targets of the highs export group, the config file so that other
    # cmake-projects can link easily against highs, and the pkg-config flie so that
    # other projects can easily build against highs
    install(EXPORT highs-targets FILE highs-targets.cmake DESTINATION
      ${CMAKE_INSTALL_LIBDIR}/cmake/highs)
    install(FILES "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs-config.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/highs)
    install(FILES "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs.pc"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

  else()
    # FAST_BUILD is set to on.
    # At the moment used only for gradually updating the CMake targets build and
    # install / export.
    # Define library in modern CMake using target_*()
    # No interfaces (apart from c); No ipx; New (short) ctest instances.
    add_library(highs)
    add_library(${PROJECT_NAMESPACE}::highs ALIAS highs)

    if(${BUILD_SHARED_LIBS})
      # put version information into shared library file
      set_target_properties(highs PROPERTIES
        VERSION
        ${HIGHS_VERSION_MAJOR}.${HIGHS_VERSION_MINOR}.${HIGHS_VERSION_PATCH}
        SOVERSION ${HIGHS_VERSION_MAJOR}.${HIGHS_VERSION_MINOR})
    endif()

    set_target_properties(highs PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(APPLE)
      set_target_properties(highs PROPERTIES
        INSTALL_RPATH "@loader_path")
    endif()

    target_include_directories(highs PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${HIGHS_BINARY_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/highs>
    )

    target_include_directories(highs PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interfaces>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/io>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pdlp>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pdlp/cupdlp>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipm>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipm/ipx>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipm/basiclu>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lp_data>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mip>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/model>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/parallel>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/presolve>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qpsolver>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simplex>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/util>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>)

    target_include_directories(highs PRIVATE
      $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/>
      $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/filereader>
      $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/pdqsort>)

    target_sources(highs PRIVATE ${sources} ${headers})

    if(ZLIB AND ZLIB_FOUND)
      target_include_directories(highs PRIVATE
        $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/zstr>
      )
      target_link_libraries(highs ZLIB::ZLIB)
      set(CONF_DEPENDENCIES "include(CMakeFindDependencyMacro)\nfind_dependency(ZLIB)")
    endif()

    # # on UNIX system the 'lib' prefix is automatically added
    # set_target_properties(highs PROPERTIES
    # OUTPUT_NAME "highs"
    # MACOSX_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

    # if (UNIX)
    # set_target_properties(highs PROPERTIES
    # LIBRARY_OUTPUT_DIRECTORY "${HIGHS_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    # endif()
    
    # set_target_properties(highs PROPERTIES PUBLIC_HEADER "src/Highs.h;src/lp_data/HighsLp.h;src/lp_data/HighsLpSolverObject.h")

    # install the header files of highs
    foreach(file ${headers})
      get_filename_component(dir ${file} DIRECTORY)

      if(NOT dir STREQUAL "")
        string(REPLACE ../extern/ "" dir ${dir})
      endif()

      install(FILES ${file} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs/${dir})
    endforeach()
    install(FILES ${HIGHS_BINARY_DIR}/HConfig.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs)

    # Configure the config windows version file
    configure_file(${HIGHS_SOURCE_DIR}/version.rc.in
      "${HIGHS_BINARY_DIR}/version.rc" @ONLY)
    if(MSVC)
      set(win_version_file ${HIGHS_BINARY_DIR}/version.rc)
    else()
      set(win_version_file)
    endif()

    # target_compile_options(highs PRIVATE "-Wall")
    # target_compile_options(highs PRIVATE "-Wunused")
    target_sources(highs PRIVATE ${basiclu_sources} ${ipx_sources} ${cupdlp_sources} 
        ipm/IpxWrapper.cpp pdlp/CupdlpWrapper.cpp ${win_version_file})
    
    if (UNIX)
      target_compile_options(highs PRIVATE "-Wno-unused-variable")
      target_compile_options(highs PRIVATE "-Wno-unused-const-variable")
    endif()

    if (BUILD_CXX)
    # Configure the config file for the build tree:
    # Either list all the src/* directories here, or put explicit paths in all the
    # include statements.
    # M reckons that the latter is more transparent, and I'm inclined to agree.
    set(CONF_INCLUDE_DIRS "${HIGHS_SOURCE_DIR}/src" "${HIGHS_BINARY_DIR}")
    configure_file(${HIGHS_SOURCE_DIR}/highs-config.cmake.in
      "${HIGHS_BINARY_DIR}/highs-config.cmake" @ONLY)

    # Configure the config file for the install
    set(CONF_INCLUDE_DIRS "\${CMAKE_CURRENT_LIST_DIR}/../../../${CMAKE_INSTALL_INCLUDEDIR}")
    configure_file(${HIGHS_SOURCE_DIR}/highs-config.cmake.in
      "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs-config.cmake" @ONLY)

    # Configure the pkg-config file for the install
    configure_file(${HIGHS_SOURCE_DIR}/highs.pc.in
      "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs.pc" @ONLY)

    # Install the targets of the highs export group, the config file so that other
    # cmake-projects can link easily against highs, and the pkg-config flie so that
    # other projects can easily build against highs
    install(FILES "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs-config.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/highs)
    install(FILES "${HIGHS_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/highs.pc"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
      
  endif()
  endif()

  if(FORTRAN_FOUND)
    set(fortransources interfaces/highs_fortran_api.f90)
    set(CMAKE_Fortran_MODULE_DIRECTORY ${HIGHS_BINARY_DIR}/modules)
    add_library(FortranHighs interfaces/highs_fortran_api.f90)

    if(NOT FAST_BUILD)
      target_link_libraries(FortranHighs PUBLIC libhighs)
    else()
      target_link_libraries(FortranHighs PUBLIC highs)
    endif()

    install(TARGETS FortranHighs 
      LIBRARY
      ARCHIVE
      RUNTIME
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs
      MODULES DESTINATION modules)
    if(NOT MSVC)
      install(FILES ${HIGHS_BINARY_DIR}/modules/highs_fortran_api.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs/fortran)
    else()
      install(FILES ${HIGHS_BINARY_DIR}/modules/${CMAKE_BUILD_TYPE}/highs_fortran_api.mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs/fortran)
    endif()
    # use link rpath
    # set_target_properties(FortranHighs PROPERTIES INSTALL_RPATH
    #   "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  endif(FORTRAN_FOUND)

  if(CSHARP_FOUND)
    message(STATUS "CSharp supported")
    set(csharpsources
      interfaces/highs_csharp_api.cs)
    add_library(HighsCsharp SHARED interfaces/highs_csharp_api.cs)
    target_compile_options(HighsCsharp PUBLIC "/unsafe")
    add_executable(csharpexample ../examples/call_highs_from_csharp.cs)
    target_compile_options(csharpexample PUBLIC "/unsafe")
    target_link_libraries(csharpexample PUBLIC HighsCsharp)
  else()
    message(STATUS "No CSharp support")
  endif()

  find_package(Threads)
  if(Threads_FOUND)
    include(CheckAtomic)
    if(HAVE_CXX_ATOMICS64_WITH_LIB)
      if(FAST_BUILD)
        target_link_libraries(highs atomic)
      else()
        target_link_libraries(libhighs atomic)
      endif()
    endif()
  endif()

  if(FAST_BUILD)
    target_link_libraries(highs Threads::Threads)
  else()
    target_link_libraries(libhighs Threads::Threads)
  endif()

endif()

# if (BUILD_PYTHON_SETUP)

#   set(CMAKE_VERBOSE_MAKEFILE ON)

#   # Find Python 3

#   find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
#   find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

#   include(FetchContent)

#   message(CHECK_START "Fetching pybind11")
#   list(APPEND CMAKE_MESSAGE_INDENT "  ")
#   set(PYBIND11_INSTALL ON)
#   set(PYBIND11_TEST OFF)
#   FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
#     GIT_TAG "v2.11.1"
#   )
#   FetchContent_MakeAvailable(pybind11)
#   list(POP_BACK CMAKE_MESSAGE_INDENT)
#   message(CHECK_PASS "fetched")

#   function(search_python_module)
#     set(options NO_VERSION)
#     set(oneValueArgs NAME PACKAGE)
#     set(multiValueArgs "")
#     cmake_parse_arguments(MODULE
#       "${options}"
#       "${oneValueArgs}"
#       "${multiValueArgs}"
#       ${ARGN}
#     )
#     message(STATUS "Searching python module: \"${MODULE_NAME}\"")
#     if(${MODULE_NO_VERSION})
#       execute_process(
#         COMMAND ${Python3_EXECUTABLE} -c "import ${MODULE_NAME}"
#         RESULT_VARIABLE _RESULT
#         ERROR_QUIET
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#       )
#       set(MODULE_VERSION "unknown")
#     else()
#       execute_process(
#         COMMAND ${Python3_EXECUTABLE} -c "import ${MODULE_NAME}; print(${MODULE_NAME}.__version__)"
#         RESULT_VARIABLE _RESULT
#         OUTPUT_VARIABLE MODULE_VERSION
#         ERROR_QUIET
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#       )
#     endif()
#     if(${_RESULT} STREQUAL "0")
#       message(STATUS "Found python module: \"${MODULE_NAME}\" (found version \"${MODULE_VERSION}\")")
#     else()
#         message(FATAL_ERROR "Can't find python module: \"${MODULE_NAME}\", please install it using your system package manager.")
#     endif()
#   endfunction()

#   search_python_module(
#     NAME setuptools
#     PACKAGE setuptools)
#   search_python_module(
#     NAME wheel
#     PACKAGE wheel)

#   set(PYTHON_PROJECT "highspy")
#   message(STATUS "Python project: ${PYTHON_PROJECT}")
#   set(PYTHON_PROJECT_DIR ${PROJECT_BINARY_DIR}/${PYTHON_PROJECT})
#   message(STATUS "Python project build path: ${PYTHON_PROJECT_DIR}")

#   pybind11_add_module(highspy ../highspy/highs_bindings.cpp)

#   target_compile_definitions(highspy 
#                   PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

#   target_include_directories(highspy PUBLIC ${include_dirs})

#   target_sources(highspy PUBLIC ${sources} ${headers})
#   add_library(${PROJECT_NAMESPACE}::highspy ALIAS highspy)

#   target_compile_definitions(highspy
#                             PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# endif()