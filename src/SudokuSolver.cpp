#include "../include/SudokuSolver.hpp"

// Can `num` be placed at (row, col) without breaking Sudoku rules?
bool SudokuSolver::isValid(int grid[9][9], int row, int col, int num) {
    for (int i = 0; i < 9; ++i) {
        if (grid[row][i] == num) return false;  // row
        if (grid[i][col] == num) return false;  // column
    }
    int boxRow = (row / 3) * 3;
    int boxCol = (col / 3) * 3;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            if (grid[boxRow + r][boxCol + c] == num) return false;  // 3x3 box
    return true;
}

// Standard backtracking. Returns true and fills `grid` in place on success.
bool SudokuSolver::backtrack(int grid[9][9]) {
    if (++steps > MAX_STEPS) return false;  // give up on inconsistent input
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            if (grid[row][col] != 0) continue;
            for (int num = 1; num <= 9; ++num) {
                if (isValid(grid, row, col, num)) {
                    grid[row][col] = num;
                    if (backtrack(grid)) return true;
                    grid[row][col] = 0;  // undo
                }
            }
            return false;  // no digit fits this empty cell -> dead end
        }
    }
    return true;  // no empty cell left -> solved
}

bool SudokuSolver::solve(int grid[9][9]) {
    steps = 0;
    return backtrack(grid);
}
