#pragma once
#include <vector>

class SudokuSolver {
private:
    bool isValid(int grid[9][9], int row, int col, int num);

public:
    bool solve(int grid[9][9]); 
};