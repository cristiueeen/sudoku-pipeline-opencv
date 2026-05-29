#pragma once
#include <opencv2/opencv.hpp>
#include "DigitRecognizer.hpp"
#include "SudokuSolver.hpp"

class SudokuPipeline {
private:
    DigitRecognizer recognizer;
    SudokuSolver solver;

    cv::Mat extractAndWarpGrid(const cv::Mat& original, std::array<cv::Point2f, 4>& outCorners);
    void extractDigits(const cv::Mat& warpedGrid, int gridData[9][9]);
    cv::Mat createResultOverlay(int originalGrid[9][9], int solvedGrid[9][9]);

public:
    SudokuPipeline();

    cv::Mat processImage(const cv::Mat& inputImage);
};