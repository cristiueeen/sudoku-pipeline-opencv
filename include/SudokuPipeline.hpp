#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <vector>
#include "algorithms.hpp"
#include "DigitRecognizer.hpp"
#include "SudokuSolver.hpp"


struct PipelineResult {
    bool gridFound = false;                 
    bool solved    = false;                 

    int recognised[9][9] = {{0}};           
    int solution[9][9]   = {{0}};           

    cv::Mat thresholded;                    
    cv::Mat warpedClean;                
    cv::Mat output;                     

    std::vector<algorithms::CellDigit> cells;  
};


class SudokuPipeline {
private:
    static constexpr int WARP = 450;        

    DigitRecognizer recognizer;
    SudokuSolver    solver;
    cv::Mat         lastMinv;               

    bool locateAndWarp(const cv::Mat& gray, cv::Mat& thresholded,
                       cv::Mat& warpedClean, cv::Mat& M, cv::Mat& Minv);

   
    cv::Mat drawSolutionBack(const cv::Mat& inputColor, const cv::Mat& Minv,
                             const int recognised[9][9], const int solution[9][9]);

public:
    SudokuPipeline();

    PipelineResult process(const cv::Mat& inputGray, const cv::Mat& inputColor);

    int harvest(const PipelineResult& res, const std::string& sourceName);
};
