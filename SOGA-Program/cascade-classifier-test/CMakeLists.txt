cmake_minimum_required(VERSION 3.5)
project(cascade_classifier_test)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(cascade_classifier_test_cpu main_cpu.cpp)
target_link_libraries(cascade_classifier_test_cpu ${OpenCV_LIBS})

add_executable(cascade_classifier_test_gpu main_gpu.cpp)
target_link_libraries(cascade_classifier_test_gpu ${OpenCV_LIBS})
