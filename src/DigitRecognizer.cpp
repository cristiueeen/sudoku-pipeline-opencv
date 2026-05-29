#include <numeric>
#include <limits>
#include <stdexcept>
#include "../include/DigitRecognizer.hpp"


cv::Mat DigitRecognizer::normalise(const cv::Mat& cell, int size) {
    cv::Mat resized;
    cv::resize(cell, resized, {size, size});
    return resized;
    
}


std::vector<float> DigitRecognizer::getHorizontalProjection(const cv::Mat& cell) {
    cv::Mat norm = normalise(cell);
    std::vector<float> proj(norm.rows, 0.f);

    for (int r = 0; r < norm.rows; ++r)
        for (int c = 0; c < norm.cols; ++c)
            if (norm.at<uchar>(r, c) >= 128)
                proj[r] += 1.f;

    float maxVal = *std::max_element(proj.begin(), proj.end());
    if (maxVal > 0)
        for (float& v : proj) v /= maxVal;

    return proj;
}

std::vector<float> DigitRecognizer::getVerticalProjection(const cv::Mat& cell) {
    cv::Mat norm = normalise(cell);
    std::vector<float> proj(norm.cols, 0.f);

    for (int r = 0; r < norm.rows; ++r)
        for (int c = 0; c < norm.cols; ++c)
            if (norm.at<uchar>(r, c) >= 128)
                proj[c] += 1.f;

    float maxVal = *std::max_element(proj.begin(), proj.end());
    if (maxVal > 0)
        for (float& v : proj) v /= maxVal;

    return proj;
}


float DigitRecognizer::calculateDistance(const DigitFeatures& f1, const DigitFeatures& f2) {
    auto sse = [](const std::vector<float>& a, const std::vector<float>& b) {
        float dist = 0.f;
        for (int i = 0; i < (int)a.size(); ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return dist;
    };

    return sse(f1.h_proj, f2.h_proj) + sse(f1.v_proj, f2.v_proj);
}

DigitRecognizer::DigitRecognizer() {
    for (int digit = 1; digit <= 9; ++digit) {
        std::string path = "templates/digit_" + std::to_string(digit) + ".png";
        cv::Mat tmpl = cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (tmpl.empty()) {
            throw std::runtime_error("DigitRecognizer: missing template " + path);
        }

        DigitFeatures features;
        features.h_proj = getHorizontalProjection(tmpl);
        features.v_proj = getVerticalProjection(tmpl);
        referenceDictionary[digit] = features;
    }
}


int DigitRecognizer::recognizeDigit(const cv::Mat& cell) {
    DigitFeatures query;
    query.h_proj = getHorizontalProjection(cell);
    query.v_proj = getVerticalProjection(cell);

    float bestDist  = std::numeric_limits<float>::max();
    int   bestDigit = 0;

    for (const auto& [digit, tmplFeatures] : referenceDictionary) {
        float dist = calculateDistance(query, tmplFeatures);
        if (dist < bestDist) {
            bestDist  = dist;
            bestDigit = digit;
        }
    }
    return bestDigit;
}