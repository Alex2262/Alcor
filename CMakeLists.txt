cmake_minimum_required(VERSION 3.21)
project(Alcor)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

#set(GCC_COVERAGE_LINK_FLAGS "-static")

#set(CMAKE_EXE_LINKER_FLAGS "${GCC_COVERAGE_LINK_FLAGS}")
include_directories(.)

add_executable(Alcor
        src/constants.h
        src/main.cpp
        src/move.cpp
        src/move.h
        src/perft.cpp
        src/perft.h
        src/position.cpp
        src/position.h
        src/useful.cpp
        src/useful.h
        src/uci.cpp
        src/uci.h
        src/zobrist.h
        src/see.cpp
        src/see.h
        src/types.h
        src/bitboard.cpp
        src/bitboard.h
        src/attacks.h
        src/fixed_vector.h
        src/nnue.cpp
        src/nnue.h
        src/incbin.h
        src/mcts.h
        src/mcts.cpp)

target_link_libraries(Alcor Threads::Threads)