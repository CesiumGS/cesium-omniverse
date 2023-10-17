# replace a string in the utils header with the intended working dir for the test executable
set(TEST_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/testUtils.h.in" "${CMAKE_CURRENT_LIST_DIR}/testUtils.h")
