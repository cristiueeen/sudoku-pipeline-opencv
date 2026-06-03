#include <opencv2/opencv.hpp>
#include <cstring>
#include <filesystem>
#include <iostream>
#include "../include/SudokuPipeline.hpp"

static void printGrid(const int g[9][9]) {
    std::cout << "+-------+-------+-------+\n";
    for (int r = 0; r < 9; ++r) {
        std::cout << "| ";
        for (int c = 0; c < 9; ++c) {
            std::cout << (g[r][c] == 0 ? '.' : char('0' + g[r][c])) << ' ';
            if ((c + 1) % 3 == 0) std::cout << "| ";
        }
        std::cout << "\n";
        if ((r + 1) % 3 == 0) std::cout << "+-------+-------+-------+\n";
    }
}

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <image-path> [--headless] [--harvest]\n"
              << "  <image-path>  path to a Sudoku image (required)\n"
              << "  --headless    do not open any windows\n"
              << "  --harvest     save correctly-read digits to templates_knn/\n";
}

int main(int argc, char** argv) {
    std::string imagePath;
    bool headless = false;
    bool harvest  = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--headless") == 0) headless = true;
        else if (std::strcmp(argv[i], "--harvest") == 0) { harvest = true; headless = true; }
        else if (argv[i][0] == '-') { usage(argv[0]); return 2; }
        else if (imagePath.empty()) imagePath = argv[i];
        else { usage(argv[0]); return 2; }
    }
    if (imagePath.empty()) { usage(argv[0]); return 2; }

    auto show = [&](const std::string& name, const cv::Mat& m) {
        if (headless || m.empty()) return;
        cv::imshow(name, m);
        cv::waitKey(0);
    };

    cv::Mat inputColor = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat inputGray  = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (inputGray.empty()) {
        std::cerr << "Error: could not open image '" << imagePath << "'\n";
        return -1;
    }
    show("Initial Sudoku", inputGray);

    SudokuPipeline pipeline;
    PipelineResult res = pipeline.process(inputGray, inputColor);

    if (!res.gridFound) {
        std::cerr << "Error: no grid found in '" << imagePath << "'\n";
        return -1;
    }
    show("Thresholded", res.thresholded);
    show("Warped Grid", res.warpedClean);

    std::cout << "\nRecognised grid:\n";
    printGrid(res.recognised);

    if (!res.solved) {
        std::cerr << "\nCould not solve the puzzle (likely a digit was "
                     "misread). Aborting overlay.\n";
        return 1;
    }
    std::cout << "\nSolved grid:\n";
    printGrid(res.solution);

    if (harvest) {
        std::string base = std::filesystem::path(imagePath).stem().string();
        int saved = pipeline.harvest(res, base);
        std::cout << "\nHarvested " << saved << " digit samples to templates_knn/\n";
        return 0;
    }

    cv::imwrite("result.png", res.output);
    std::cout << "\nWrote result.png\n";
    show("Solved Sudoku", res.output);
    return 0;
}
