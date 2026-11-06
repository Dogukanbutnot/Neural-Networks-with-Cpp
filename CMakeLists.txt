cmake_minimum_required(VERSION 3.15)
project(nn_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Derleme turu verilmediyse Release sec.
# NOT: VS Code CMake Tools varsayilan olarak Debug secebilir; Debug'ta egitim
# cok yavas calisir. Alttaki mavi bardan "Release" sec.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Derleme turu" FORCE)
endif()

option(NN_USE_OPENMP "Buyuk matris carpimlarini OpenMP ile paralellestir" ON)
option(NN_NO_SIMD    "SIMD (AVX2/SSE2) cekirdeklerini kapat, sade C++ kullan" OFF)
option(NN_MSVC_AVX2  "MSVC icin /arch:AVX2 ile derle (AVX2 desteginiz varsa)" OFF)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_DEBUG   "-g -Wall -Wextra")
endif()

add_executable(nn    main.cpp  matrix.cpp network.cpp)
add_executable(bench bench.cpp matrix.cpp)
add_executable(sched_test scheduler_test.cpp matrix.cpp network.cpp)
add_executable(opt_test optimizer_test.cpp matrix.cpp network.cpp)
add_executable(grad_test gradcheck_test.cpp matrix.cpp network.cpp)
add_executable(es_test earlystop_test.cpp matrix.cpp network.cpp)
add_executable(conv_test conv_test.cpp matrix.cpp network.cpp)

find_package(OpenMP)
foreach(tgt nn bench sched_test opt_test grad_test es_test conv_test)
    if(NN_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(NN_NO_SIMD)
        target_compile_definitions(${tgt} PRIVATE NN_NO_SIMD)
    endif()
    if(MSVC AND NN_MSVC_AVX2)
        target_compile_options(${tgt} PRIVATE /arch:AVX2)
    endif()
endforeach()

if(NN_USE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: acik")
else()
    message(STATUS "OpenMP: kapali")
endif()
