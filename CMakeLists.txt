cmake_minimum_required(VERSION 3.16)
project(ImageFilters LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} found in ${OpenCV_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# The filters
add_library(filters_core STATIC filters.cpp filters.h)
target_include_directories(filters_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(filters_core PUBLIC opencv_core opencv_imgproc)

# Command-line program
add_executable(filters main.cpp)
target_link_libraries(filters PRIVATE filters_core opencv_imgcodecs opencv_highgui)

option(BUILD_TESTING "Build the tests" ON)
if(BUILD_TESTING)
  enable_testing()
  set(SAMPLE_IMAGE ${CMAKE_CURRENT_SOURCE_DIR}/bicycle.bmp)
  set(TEST_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_output)

  # The sample images are read from the source directory
  add_executable(filters_tests tests/test_filters.cpp)
  target_link_libraries(filters_tests PRIVATE filters_core opencv_imgcodecs)
  target_compile_definitions(filters_tests PRIVATE SAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  foreach(name IN ITEMS kernels reflect_index centred_window step_edges flat_images box_filter_all_sums
                        box_filter_samples box_filter_random sobel_samples sobel_random display filters_by_key
                        grid arguments)
    add_test(NAME ${name} COMMAND filters_tests ${name})
  endforeach()

  # The program itself, without a display. The error cases are recognised by
  # their messages.
  add_test(NAME cli_no_gui COMMAND filters --no-gui --size 7 --out ${TEST_OUTPUT} ${SAMPLE_IMAGE})
  add_test(NAME cli_missing_image COMMAND filters --no-gui --out ${TEST_OUTPUT} missing.png)
  add_test(NAME cli_bad_size COMMAND filters --no-gui --size 4 ${SAMPLE_IMAGE})
  add_test(NAME cli_unknown_option COMMAND filters --no-gui --sise 5 ${SAMPLE_IMAGE})
  set_tests_properties(cli_missing_image PROPERTIES PASS_REGULAR_EXPRESSION "Could not read the image")
  set_tests_properties(cli_bad_size PROPERTIES PASS_REGULAR_EXPRESSION "--size must be 3, 5 or 7")
  set_tests_properties(cli_unknown_option PROPERTIES PASS_REGULAR_EXPRESSION "Unknown option: --sise")

  # A 2x2 image (smaller than every window), and an output path that is a file
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tiny.pgm "P2\n2 2\n255\n0 60\n120 255\n")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/not_a_directory "")
  add_test(NAME cli_tiny_image
           COMMAND filters --no-gui --size 7 --out ${TEST_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/tiny.pgm)
  add_test(NAME cli_bad_output_dir
           COMMAND filters --no-gui --out ${CMAKE_CURRENT_BINARY_DIR}/not_a_directory ${SAMPLE_IMAGE})
  set_tests_properties(cli_bad_output_dir PROPERTIES PASS_REGULAR_EXPRESSION "Could not create")
endif()
