#include "../include/SudokuPipeline.hpp"
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

SudokuPipeline::SudokuPipeline() = default;

bool SudokuPipeline::locateAndWarp(const cv::Mat& gray, cv::Mat& thresholded,
                                   cv::Mat& warpedClean, cv::Mat& M, cv::Mat& Minv) {
    cv::Mat blurred = algorithms::gaussianBlur(gray, 5);
    thresholded     = algorithms::adaptiveThreshold(blurred, 21, 15);
    cv::Mat morphed = algorithms::morphClose(thresholded, 5);

    std::vector<cv::Point> largestBlob = algorithms::findLargestBlob(morphed);
    if (largestBlob.empty()) return false;

    std::array<cv::Point2f, 4> corners = algorithms::getGridCorners(largestBlob);

    std::vector<cv::Point2f> dst = {
        {0, 0}, {(float)WARP, 0}, {(float)WARP, (float)WARP}, {0, (float)WARP}};
    std::vector<cv::Point2f> src = {corners[0], corners[1], corners[2], corners[3]};

    M    = cv::getPerspectiveTransform(src, dst);
    Minv = cv::getPerspectiveTransform(dst, src);

    cv::warpPerspective(thresholded, warpedClean, M, cv::Size(WARP, WARP),
                        cv::INTER_NEAREST);
    return true;
}

cv::Mat SudokuPipeline::drawSolutionBack(const cv::Mat& inputColor,
                                         const cv::Mat& Minv,
                                         const int recognised[9][9],
                                         const int solution[9][9]) {
    const int cell = WARP / 9;
    cv::Mat overlayWarp = cv::Mat::zeros(WARP, WARP, CV_8UC3);

    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 9; ++c) {
            if (recognised[r][c] != 0) continue;  
            std::string text = std::to_string(solution[r][c]);
            int baseline = 0;
            double scale = 1.0;
            int thick = 2;
            cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                          scale, thick, &baseline);
            cv::Point org(c * cell + (cell - ts.width) / 2,
                          r * cell + (cell + ts.height) / 2);
            cv::putText(overlayWarp, text, org, cv::FONT_HERSHEY_SIMPLEX,
                        scale, cv::Scalar(0, 180, 0), thick, cv::LINE_AA);
        }
    }

    cv::Mat overlayOrig;
    cv::warpPerspective(overlayWarp, overlayOrig, Minv, inputColor.size());

    cv::Mat result = inputColor.clone();
    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            cv::Vec3b o = overlayOrig.at<cv::Vec3b>(y, x);
            if (o[0] || o[1] || o[2]) result.at<cv::Vec3b>(y, x) = o;
        }
    }
    return result;
}

PipelineResult SudokuPipeline::process(const cv::Mat& inputGray, const cv::Mat& inputColor) {
    PipelineResult res;

    cv::Mat M, Minv;
    if (!locateAndWarp(inputGray, res.thresholded, res.warpedClean, M, Minv))
        return res;  
    res.gridFound = true;
    lastMinv = Minv;

    res.cells = algorithms::extractDigits(res.warpedClean, WARP);
    for (const auto& d : res.cells)
        res.recognised[d.row][d.col] = recognizer.recognizeDigit(d.cellImage);

    std::memcpy(res.solution, res.recognised, sizeof(res.solution));
    res.solved = solver.solve(res.solution);

    if (res.solved)
        res.output = drawSolutionBack(inputColor, Minv, res.recognised, res.solution);

    return res;
}

int SudokuPipeline::harvest(const PipelineResult& res,
                            const std::string& sourceName) {
    if (!res.solved) return 0;  

    fs::create_directories("templates_knn");
    int saved = 0;
    for (const auto& d : res.cells) {
        int val = res.recognised[d.row][d.col];
        if (val == 0 || d.cellImage.empty()) continue;
        std::string fn = "templates_knn/digit_" + std::to_string(val) + "_" +
                         sourceName + "_" + std::to_string(d.row) +
                         std::to_string(d.col) + ".png";
        cv::imwrite(fn, d.cellImage);
        ++saved;
    }
    return saved;
}
