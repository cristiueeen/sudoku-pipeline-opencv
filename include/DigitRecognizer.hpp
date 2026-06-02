#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

// Feature vector for a single normalised digit image.
//  - pixels : flattened S*S binary overlap map (0/1)
//  - zoning : Z*Z grid of foreground densities (captures spatial layout,
//             the key to telling 1/7 and 3/8 apart)
//  - h_proj : per-row foreground density (S values)
//  - v_proj : per-column foreground density (S values)
struct DigitFeatures {
    std::vector<float> pixels;
    std::vector<float> zoning;
    std::vector<float> h_proj;
    std::vector<float> v_proj;
    bool valid = false;
};

class DigitRecognizer {
private:
    // Canonical sizes (validated offline against real warped cells).
    static constexpr int S      = 28;  // normalised canvas side
    static constexpr int MARGIN = 4;   // empty border kept around the digit
    static constexpr int Z      = 7;   // zoning grid is Z x Z
    static constexpr int K      = 5;   // neighbours for the k-NN vote

    // Every labelled reference sample: the canonical hand-made templates plus
    // any real digits harvested from solved grids (templates_knn/).
    std::vector<std::pair<int, DigitFeatures>> references;

    cv::Mat normalise(const cv::Mat& cell);
    DigitFeatures extractFeatures(const cv::Mat& normalised);
    float calculateDistance(const DigitFeatures& a, const DigitFeatures& b);
    void addSample(int digit, const cv::Mat& image);

public:
    // Builds the reference set from templates_knn/ (the harvested real digits).
    // If a legacy templates/ folder is present it is also loaded as extra
    // seeds, but it is not required.
    DigitRecognizer();

    // Returns the recognised digit (1-9), or 0 for an empty cell.
    int recognizeDigit(const cv::Mat& cell);

    // Number of reference samples loaded (for diagnostics).
    std::size_t sampleCount() const { return references.size(); }
};
