#pragma once
#include <vector>

class SudokuSolver {
private:
    long long steps = 0;
    static constexpr long long MAX_STEPS = 2'000'000;

    bool isValid(int grid[9][9], int row, int col, int num);
    bool backtrack(int grid[9][9]);

public:
    bool solve(int grid[9][9]);
};
