cmake_minimum_required(VERSION 2.8)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( DisplayImage my.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

