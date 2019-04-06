cmake_minimum_required(VERSION 3.5)
project(face_detection)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(face_detection_cpu main_cpu.cpp)
target_link_libraries(face_detection_cpu ${OpenCV_LIBS} realsense2)

add_executable(face_detection_gpu main_gpu.cpp)
target_link_libraries(face_detection_gpu ${OpenCV_LIBS} realsense2)
