cmake_minimum_required(VERSION 3.10)

project(floorplan)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add all .cpp files in the src directory to the project
file(GLOB SOURCES "src/*.cpp")

# Include the src directory
include_directories("src")

# Generate the executable
add_executable(floorplan_exec ${SOURCES})
