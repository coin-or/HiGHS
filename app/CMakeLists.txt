# create highs binary using library without pic
add_executable(highs RunHighs.cpp ${HIGHS_SOURCE_DIR}/src/lp_data/HighsRuntimeOptions.h)
if (LINUX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)) 
    target_link_libraries(highs libhighs)
else()
    target_link_libraries(highs libhighs libipx libbasiclu)
endif()

# install the binary
install(TARGETS highs EXPORT highs-targets
        RUNTIME DESTINATION bin)
