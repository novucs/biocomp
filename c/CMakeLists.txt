cmake_minimum_required(VERSION 3.15)
project(c)

set(CMAKE_CXX_STANDARD 14)
file(GLOB source_files "src/*.h" "src/*.cpp")
add_executable(c ${source_files} src/Main.cpp src/Main.h)
