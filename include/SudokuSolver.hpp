#pragma once
#include <vector>

class SudokuSolver {
private:
    // Guard against pathological backtracking when the recognised grid is
    // inconsistent: bail out after this many search steps.
    long long steps = 0;
    static constexpr long long MAX_STEPS = 2'000'000;

    bool isValid(int grid[9][9], int row, int col, int num);
    bool backtrack(int grid[9][9]);

public:
    // Returns true and fills `grid` on success; false if unsolvable or the
    // step budget is exhausted.
    bool solve(int grid[9][9]);
};
