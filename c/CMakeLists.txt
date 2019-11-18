cmake_minimum_required(VERSION 3.15)
project(rulebasedga)

set(CMAKE_CXX_STANDARD 17)
file(GLOB source_files "src/*.h" "src/*.cpp")
add_executable(rulebasedga ${source_files})
target_link_libraries(rulebasedga -lpthread)
