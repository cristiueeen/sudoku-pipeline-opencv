cmake_minimum_required(VERSION 4.2.1)
project(sudoku-project C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# Gather all source files from the src directory
file(GLOB SOURCES "src/*.cpp")

add_executable(sudoku-app ${SOURCES})

# Add the include folder to the include directories
target_include_directories(sudoku-app PRIVATE ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sudoku-app PRIVATE ${OpenCV_LIBS})