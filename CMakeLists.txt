cmake_minimum_required(VERSION 3.10)
project(calibrate)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibrate main.cpp)
target_link_libraries( calibrate ${OpenCV_LIBS} )