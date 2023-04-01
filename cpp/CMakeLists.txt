cmake_minimum_required(VERSION 3.5)

project(HHD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(HHD main.cpp)

install(TARGETS HHD
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

target_include_directories(HHD PUBLIC "D:/dev/eigen-3.4.0/" )
target_include_directories(HHD PUBLIC "D:/dev/libigl-2.4.0/include/")
target_include_directories(HHD PUBLIC "D:/dev/tetgen1.6.0/")
find_library(LIBTET tet "D:/dev/tetgen1.6.0/")
target_link_libraries(HHD ${LIBTET})

add_subdirectory("polyscope")
target_link_libraries(HHD polyscope)
