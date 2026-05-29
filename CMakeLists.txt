cmake_minimum_required(VERSION 3.15)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release modunda optimizasyon aç, Debug modunda debug bilgisi ekle
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -Wall -Wextra")

add_executable(nn
    main.cpp
    matrix.cpp
    network.cpp
)
