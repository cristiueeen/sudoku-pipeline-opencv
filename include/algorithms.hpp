#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

namespace algorithms {
    struct CellDigit {
        int row, col;
        cv::Mat cellImage;
    };
    struct Blob {
        std::vector<cv::Point> pixels;
        cv::Rect boundingBox;
        cv::Point2f center;
        int area;
    };

   struct BlobFilterParams {
        float minAreaFraction  = 0.02f;
        float maxAreaFraction  = 0.80f;
        float maxAspectRatio   = 1.5f;
        float minAspectRatio   = 0.2f;
        int   cropPadding      = 0;   
    };
    cv::Mat gaussianBlur(const cv::Mat& src, int kernel_size = 5);
    cv::Mat adaptiveThreshold(const cv::Mat& src, int windowSize, int c);
    cv::Mat morphClose(const cv::Mat& src, int kernelSize = 3);
    cv::Mat morphClose(const cv::Mat& src, int kWidth, int kHeight);

    std::vector<cv::Point> findLargestBlob(const cv::Mat& binaryImg);
    std::array<cv::Point2f, 4> getGridCorners(const std::vector<cv::Point>& blobPixels);
    
    std::vector<float> getHorizontalProjection(const cv::Mat& cellBlob);
    std::vector<float> getVerticalProjection(const cv::Mat& cellBlob);
    cv::Mat_<uchar> adaptiveThreshold(const cv::Mat_<uchar>& src, int windowSize = 21, int C = 1000);
    std::vector<CellDigit> extractDigits(const cv::Mat& warpedBinary, int warpSize, const BlobFilterParams& params = {});
}