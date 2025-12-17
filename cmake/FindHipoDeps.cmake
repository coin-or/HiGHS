# Fetch OpenBLAS
if (BUILD_OPENBLAS)
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    # set(BUILD_SHARED_LIBS ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(BUILD_TESTING OFF)
    set(CMAKE_Fortran_COMPILER OFF)

    # Define the size-minimizing flags as a list
    set(OPENBLAS_MINIMAL_FLAGS
        # Exclude components not used by HiGHS
        -DNO_LAPACK=ON
        -DNO_LAPACKE=ON
        -DNO_COMPLEX=ON
        -DNO_SINGLE=ON
        -DONLY_BLAS=ON
    )

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8|arm")
        message(STATUS "ARM architecture detected. Applying -DTARGET=ARMV8.")
        list(APPEND OPENBLAS_MINIMAL_FLAGS -DTARGET=ARMV8)
    endif()

    # CMAKE_SIZEOF_VOID_P is 4 for 32-bit builds, 8 for 64-bit builds.
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
        message(STATUS "32-bit target detected. Applying 32-bit configuration flags for OpenBLAS.")

        set(OPENBLAS_WIN_32 ON)

        list(APPEND OPENBLAS_MINIMAL_FLAGS -DCMAKE_GENERATOR_PLATFORM=Win32)

        # Crucial for static linking: Force OpenBLAS to use the static runtime
        if (NOT BUILD_SHARED_LIBS)
            list(APPEND OPENBLAS_MINIMAL_FLAGS -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded)
        endif()

        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

        list(APPEND OPENBLAS_MINIMAL_FLAGS -DUSE_THREAD=OFF)
        list(APPEND OPENBLAS_MINIMAL_FLAGS -DINTERFACE64=0)


        # Note: If OpenBLAS has an internal logic flag to force 32-bit, you would add it here.
        # Example (hypothetical):
        # list(APPEND OPENBLAS_MINIMAL_FLAGS -DOPENBLAS_32BIT=ON)

        # If the MSVC runtime library issue persists, you can try this flag as well,
        # though CMAKE_GENERATOR_PLATFORM should usually be sufficient.
        # list(APPEND OPENBLAS_MINIMAL_FLAGS -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL)
    endif()

    message(CHECK_START "Fetching OpenBLAS")
    list(APPEND CMAKE_MESSAGE_INDENT "  ")
    FetchContent_Declare(
        openblas
        GIT_REPOSITORY "https://github.com/OpenMathLib/OpenBLAS.git"
        GIT_TAG        "v0.3.30"
        GIT_SHALLOW TRUE
        UPDATE_COMMAND git reset --hard
        CMAKE_ARGS ${OPENBLAS_MINIMAL_FLAGS}
    )
    FetchContent_MakeAvailable(openblas)
    list(POP_BACK CMAKE_MESSAGE_INDENT)
    message(CHECK_PASS "fetched")

    if (TARGET openblas)
        get_target_property(_openblas_aliased openblas ALIASED_TARGET)
        if(_openblas_aliased)
            set(_openblas_target ${_openblas_aliased})
            message(STATUS "OpenBLAS is an alias for: ${_openblas_target}")
        else()
            set(_openblas_target openblas)
        endif()
    elseif (TARGET openblas_static)
        set(_openblas_target openblas_static)
    elseif (TARGET openblas_shared)
        set(_openblas_target openblas_shared)
    else()
        message(FATAL_ERROR "OpenBLAS target not found")
    endif()
    message(STATUS "OpenBLAS target: ${_openblas_target}")

    return()
endif()

# Find BLAS
set(BLAS_ROOT "" CACHE STRING "Root directory of BLAS or OpenBLAS")
if (NOT BLAS_ROOT STREQUAL "")
    message(STATUS "BLAS_ROOT is " ${BLAS_ROOT})
endif()

set(USE_CMAKE_FIND_BLAS ON)

# Optionally set the vendor:
# set(BLA_VENDOR libblastrampoline)

if (NOT USE_CMAKE_FIND_BLAS)
    if (WIN32)
        if (NOT (BLAS_ROOT STREQUAL ""))
            message(STATUS "Looking for blas in " ${BLAS_ROOT})
            set(OpenBLAS_ROOT ${BLAS_ROOT})
            message(STATUS "OpenBLAS_ROOT is ${OpenBLAS_ROOT} ")
            find_package(OpenBLAS CONFIG NO_DEFAULT_PATH)

            if(OpenBLAS_FOUND)
                message(STATUS "OpenBLAS CMake config path: ${OpenBLAS_DIR}")
            else()
                message(STATUS "OpenBLAS not found in ${BLAS_ROOT}")
            endif()
        endif()

        if ((BLAS_ROOT STREQUAL "") OR (NOT OpenBLAS_FOUND))
            # (NOT OpenBLAS_FOUND AND NOT BLAS_FOUND))
            message(STATUS "Looking for blas")

            find_package(OpenBLAS REQUIRED)

            if(OpenBLAS_FOUND)
                if(TARGET OpenBLAS::OpenBLAS)
                    message(STATUS "OpenBLAS CMake config path: ${OpenBLAS_DIR}")
                elseif(OPENBLAS_LIB)
                    message(STATUS "Linking against OpenBLAS via raw library: ${OPENBLAS_LIB}")
                else()
                    # try blas
                    # find_package(BLAS)
                    # if (BLAS_FOUND)
                    #     message(STATUS "Using BLAS library: ${BLAS_LIBRARIES}")
                    #     message(STATUS "BLAS include dirs: ${BLAS_INCLUDE_DIRS}")
                    # else()
                    #     message(STATUS "OpenBLAS found but no target?")
                    # endif()
                endif()
            else()
                message(FATAL_ERROR "No BLAS library found")
            endif()
        endif()

    elseif(NOT APPLE)
        # LINUX

        # If a BLAS install was specified try to use it first.
        if (NOT (BLAS_ROOT STREQUAL ""))
            message(STATUS "Looking for blas in " ${BLAS_ROOT})

            find_library(OPENBLAS_LIB
                NAMES openblas
                HINTS "${BLAS_ROOT}/lib"
                NO_DEFAULT_PATH)

            if(OPENBLAS_LIB)
                message(STATUS "Found OpenBLAS library at ${OPENBLAS_LIB}")
            else()
                find_library(BLAS_LIB
                    NAMES blas
                    HINTS "${BLAS_ROOT}/lib"
                    NO_DEFAULT_PATH)

                if(BLAS_LIB)
                    message(STATUS "Found BLAS library at ${BLAS_LIB}")
                else()
                    message(STATUS "Did not find blas library at ${BLAS_ROOT}")
                    message(STATUS "Attempting default locations search")
                endif()
            endif()
        endif()
        if ((BLAS_ROOT STREQUAL "") OR (NOT OPENBLAS_LIB AND NOT BLAS_LIB))

            find_library(OPENBLAS_LIB
                NAMES openblas
                HINTS "${BLAS_ROOT}/lib")

            if(OPENBLAS_LIB)
                message(STATUS "Found OpenBLAS library at ${OPENBLAS_LIB}")
            else()
                find_library(BLAS_LIB
                    NAMES blas
                    HINTS "${BLAS_ROOT}/lib")

                if(BLAS_LIB)
                    message(STATUS "Found BLAS library at ${BLAS_LIB}")
                else()
                    message(FATAL_ERROR "No BLAS library found")
                endif()
            endif()
        endif()
    endif()
else()

    if (WIN32 AND NOT BLAS_LIBRARIES AND NOT BLA_VENDOR)
        find_package(OpenBLAS CONFIG)
        if(OpenBLAS_FOUND)
            message(STATUS "OpenBLAS CMake config path: ${OpenBLAS_DIR}")
        endif()
    endif()

    if (NOT OpenBLAS_FOUND)
        if (NOT BLA_VENDOR)
            if (APPLE)
                set (BLA_VENDOR Apple)
            elseif(LINUX)
                set (BLA_VENDOR OpenBLAS)
            elseif(WIN32)
                set (BLA_VENDOR OpenBLAS)
            endif()

            find_package(BLAS QUIET)
            if (BLAS_FOUND)
                message(STATUS "Using BLAS library: ${BLAS_LIBRARIES}")
                if (BLAS_INCLUDE_DIRS)
                    message(STATUS "BLAS include dirs: ${BLAS_INCLUDE_DIRS}")
                endif()
            else()
                unset(BLA_VENDOR)
            endif()
        else()
            message(STATUS "Specified BLA_VENDOR: ${BLA_VENDOR}")
        endif()

        if (NOT BLAS_FOUND)
            find_package(BLAS REQUIRED)
            if (BLAS_FOUND)
                message(STATUS "Using BLAS library: ${BLAS_LIBRARIES}")
                if (BLAS_INCLUDE_DIRS)
                    message(STATUS "BLAS include dirs: ${BLAS_INCLUDE_DIRS}")
                endif()
            else()
                message(FATAL_ERROR "No BLAS library found!")
            endif()
        endif()
    endif()
endif()
