cmake_minimum_required(VERSION 3.5)

project(BindMPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

add_subdirectory(benches)
add_subdirectory(bindings)
add_subdirectory(tests)