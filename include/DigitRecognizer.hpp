#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

struct DigitFeatures {
    std::vector<float> h_proj;      // horizontal projection histogram
    std::vector<float> v_proj;      // vertical projection histogram
    std::vector<float> zoneDensity; // spatial zone density grid
    cv::Mat normalizedImage;        // preprocessed 28x28 image for pixel matching
};

struct TemplateEntry {
    int digit;
    DigitFeatures features;
};

class DigitRecognizer {
private:
    static constexpr int NORM_SIZE = 28;
    static constexpr int ZONE_GRID = 5;  // 5x5 = 25 zones
    static constexpr float CONFIDENCE_THRESHOLD = 0.25f;

    std::vector<TemplateEntry> templates; // multiple templates per digit

    // Preprocessing
    cv::Mat preprocessCell(const cv::Mat& cell);

    // Feature extraction
    std::vector<float> getHorizontalProjection(const cv::Mat& normalized);
    std::vector<float> getVerticalProjection(const cv::Mat& normalized);
    std::vector<float> getZoneDensities(const cv::Mat& normalized);

    // Matching
    float getPixelMatchScore(const cv::Mat& a, const cv::Mat& b);
    float getProjectionDistance(const DigitFeatures& f1, const DigitFeatures& f2);
    float getZoneDistance(const DigitFeatures& f1, const DigitFeatures& f2);
    float getCombinedScore(const DigitFeatures& query, const TemplateEntry& tmpl);

    // Template augmentation
    DigitFeatures extractFeatures(const cv::Mat& preprocessed);
    void addTemplateVariants(int digit, const cv::Mat& baseImage);

public:
    DigitRecognizer(); 
    
    int recognizeDigit(const cv::Mat& cell); 
};