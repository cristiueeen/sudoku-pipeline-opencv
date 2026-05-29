#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include "../include/algorithms.hpp"
#include "../include/DigitRecognizer.hpp"

int main() {
    cv::Mat inputImage = cv::imread("images/sudoku-perspective-1.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) return -1;

    cv::imshow("Initial Sudoku", inputImage);
    cv::waitKey(0);

    auto bluredImg = algorithms::gaussianBlur(inputImage, 5);
    cv::imshow("blurred Sudoku", bluredImg);
    cv::waitKey(0);

    auto tresholdedImg = algorithms::adaptiveThreshold(bluredImg, 21, 15);
    cv::imshow("thresholded Sudoku", tresholdedImg);
    cv::waitKey(0);

    auto morphedImg = algorithms::morphClose(tresholdedImg, 5);
    cv::imshow("morphed closed Sudoku", morphedImg);
    cv::waitKey(0);

    std::vector<cv::Point> largestBlob = algorithms::findLargestBlob(morphedImg);
    
    if (largestBlob.empty()) {
        std::cerr << "Error: No grid found!" << std::endl;
        return -1;
    }

    std::array<cv::Point2f, 4> corners = algorithms::getGridCorners(largestBlob);

    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(0, 0),          
        cv::Point2f(450, 0),        
        cv::Point2f(450, 450),      
        cv::Point2f(0, 450)         
    };

    std::vector<cv::Point2f> srcPoints = {corners[0], corners[1], corners[2], corners[3]};

    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    cv::Mat warpedClean;
    cv::warpPerspective(tresholdedImg, warpedClean, perspectiveMatrix, cv::Size(450, 450), cv::INTER_NEAREST);
    
    cv::Mat warpedOriginal;
    cv::warpPerspective(inputImage, warpedOriginal, perspectiveMatrix, cv::Size(450, 450));

    cv::imshow("Final Warped Grid (Clean Digits)", warpedClean);
    cv::waitKey(0);

    
    auto extractedDigits = algorithms::extractDigits(warpedClean, 450);
    std::cout << "Extracted " << extractedDigits.size() << " digits." << std::endl;

    DigitRecognizer recognizer;
    int sudokuGrid[9][9] = {0};

    for (const auto& digit : extractedDigits) {
        int recognized = recognizer.recognizeDigit(digit.cellImage);
        sudokuGrid[digit.row][digit.col] = recognized;
    }

    std::cout << "\nRecognized Sudoku Grid:\n";
    std::cout << "+-------+-------+-------+\n";
    for (int r = 0; r < 9; ++r) {
        std::cout << "| ";
        for (int c = 0; c < 9; ++c) {
            if (sudokuGrid[r][c] == 0)
                std::cout << ". ";
            else
                std::cout << sudokuGrid[r][c] << " ";
            
            if ((c + 1) % 3 == 0) std::cout << "| ";
        }
        std::cout << "\n";
        if ((r + 1) % 3 == 0) {
            std::cout << "+-------+-------+-------+\n";
        }
    }

    return 0;
}