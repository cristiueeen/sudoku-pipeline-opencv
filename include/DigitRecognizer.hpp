#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

struct DigitFeatures {
    std::vector<float> pixels;
    std::vector<float> zoning;
    std::vector<float> h_proj;
    std::vector<float> v_proj;
    bool valid = false;
};

class DigitRecognizer {
private:
    static constexpr int S      = 28;  
    static constexpr int MARGIN = 4;   
    static constexpr int Z      = 7;   
    static constexpr int K      = 5;  

    std::vector<std::pair<int, DigitFeatures>> references;

    cv::Mat normalise(const cv::Mat& cell);
    DigitFeatures extractFeatures(const cv::Mat& normalised);
    float calculateDistance(const DigitFeatures& a, const DigitFeatures& b);
    void addSample(int digit, const cv::Mat& image);

public:
    DigitRecognizer();

    int recognizeDigit(const cv::Mat& cell);

    std::size_t sampleCount() const { return references.size(); }
};
