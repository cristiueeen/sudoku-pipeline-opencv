#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <vector>
#include "algorithms.hpp"
#include "DigitRecognizer.hpp"
#include "SudokuSolver.hpp"

// Everything the pipeline produces for one image. Intermediate images are kept
// so the CLI can optionally display them.
struct PipelineResult {
    bool gridFound = false;                 // grid blob + 4 corners located
    bool solved    = false;                 // recognised grid was solvable

    int recognised[9][9] = {{0}};           // digits read from the image
    int solution[9][9]   = {{0}};           // solved grid (valid only if solved)

    cv::Mat thresholded;                    // binarised input (intermediate)
    cv::Mat warpedClean;                    // 450x450 rectified grid (intermediate)
    cv::Mat output;                         // original + green solution overlay

    std::vector<algorithms::CellDigit> cells;  // extracted cells (for harvesting)
};

// Orchestrates the full Sudoku solving pipeline:
//   preprocess -> locate grid -> warp -> extract & recognise digits ->
//   solve -> draw the solution back onto the original image.
class SudokuPipeline {
private:
    static constexpr int WARP = 450;        // side of the rectified grid

    DigitRecognizer recognizer;
    SudokuSolver    solver;
    cv::Mat         lastMinv;               // inverse warp from the last run

    // Locate the grid in `gray`, rectify it to WARP x WARP and return the
    // clean binary warp. Fills the forward/inverse perspective matrices.
    // Returns false if no grid could be found.
    bool locateAndWarp(const cv::Mat& gray, cv::Mat& thresholded,
                       cv::Mat& warpedClean, cv::Mat& M, cv::Mat& Minv);

    // Render the solved-in digits (cells that were empty in `recognised`) onto
    // the original colour image using the inverse perspective transform.
    cv::Mat drawSolutionBack(const cv::Mat& inputColor, const cv::Mat& Minv,
                             const int recognised[9][9], const int solution[9][9]);

public:
    SudokuPipeline();

    // Run the whole pipeline. `inputColor` is used for the final overlay.
    PipelineResult process(const cv::Mat& inputGray, const cv::Mat& inputColor);

    // Save the correctly-read digits of a *solved* result into templates_knn/
    // to grow the k-NN reference set. Returns the number of samples written.
    int harvest(const PipelineResult& res, const std::string& sourceName);
};
