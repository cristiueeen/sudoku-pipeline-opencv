#!/bin/bash

# Define the executable path
EXECUTABLE="./build/sudoku-app"

# Check if the executable exists
if [ -f "$EXECUTABLE" ]; then
    # Run the executable, passing any arguments provided to this script
    "$EXECUTABLE" "$@"
else
    echo "Executable not found at $EXECUTABLE."
    echo "Please run ./build.sh first to build the project."
    exit 1
fi