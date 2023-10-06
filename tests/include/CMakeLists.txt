include(Macros)

glob_files(SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

get_property(ADDITIONAL_LIBRARIES GLOBAL PROPERTY NVIDIA_ADDITIONAL_LIBRARIES_PROPERTY)

# replace a string in the utils header with the intended working dir for the test executable
set(TEST_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
configure_file("${CMAKE_CURRENT_LIST_DIR}/testUtils.h.in" "${CMAKE_CURRENT_LIST_DIR}/testUtils.h")
