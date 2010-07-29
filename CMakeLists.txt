cmake_minimum_required(VERSION 2.6)
project(CastorTest)
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic ")
include_directories(${CastorTest_SOURCE_DIR}/src)
add_executable(CastorTest ${CastorTest_SOURCE_DIR}/src/main.cpp)

