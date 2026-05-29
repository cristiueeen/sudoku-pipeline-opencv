#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure the project
cmake ..

# Build the project
cmake --build .