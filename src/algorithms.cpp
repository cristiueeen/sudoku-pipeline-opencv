#include "../include/algorithms.hpp"

using namespace cv;

namespace algorithms{
    
    template <typename T>
    bool isInside(const cv::Mat_<T>& img, int r, int c) {
        return r >= 0 && r < img.rows && c >= 0 && c < img.cols;
    }

    template <typename T>
    Mat_<float> applyConvolution(const Mat_<T>& img, const Mat_<float>& kernel) {
        Mat_<float> convolved_img(img.size());
        convolved_img.setTo(0);
        
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                for (int k = 0; k < kernel.rows; k++) {
                    for (int l = 0; l < kernel.cols; l++) {
                        int r = i + k - kernel.rows / 2;
                        int c = j + l - kernel.cols / 2;
                        
                        if (isInside(img, r, c)) {
                            convolved_img(i,j) += static_cast<float>(img(r, c)) * kernel(k, l); 
                        }
                    }
                }
            }
        }
        return convolved_img;
    }
    std::pair<float, float> computeConvolutionLimits(const Mat_<float> kernel) {
        float negative_sum = 0.0f;
        float positive_sum = 0.0f;
        for (int i = 0; i < kernel.rows; i++) {
            for (int j = 0; j < kernel.cols; j++) {
                if (kernel(i, j) < 0)
                    negative_sum += kernel(i, j);
                else
                    positive_sum += kernel(i,j);
            }
        }
        return std::pair(negative_sum * 255, positive_sum * 255);
    }
    Mat_<uchar> applyRawNormalization(const Mat_<float> convolved_img, const Mat_<float> kernel){
        Mat_<uchar> normalized_img(convolved_img.size());
        normalized_img.setTo(0);
        std::pair theoretical_limits = computeConvolutionLimits(kernel);
        for (int i = 0; i < convolved_img.rows; i ++) {
            for ( int j = 0; j < convolved_img.cols; j++) {
                normalized_img(i,j) = static_cast<uchar>((convolved_img(i,j) - theoretical_limits.first) * 255 / (theoretical_limits.second - theoretical_limits.first));
            }
        }
        return normalized_img;
    }
    Mat_<float> create_gaussian_kernel(const int kernel_rows = 3, const int kernel_cols = 3) {
        Mat_<float> kernel(kernel_rows, kernel_cols);
        float sigma = (kernel_rows > 1) ? kernel_rows / 6.0 : kernel_cols / 6.0; 
        kernel.setTo(0.0);
        int kernel_x0 = kernel_rows / 2;
        int kernel_y0 = kernel_cols / 2;
        float sum = 0.0;
        for (int i = 0; i < kernel_rows; i ++) {
            for (int j = 0; j < kernel_cols; j ++) {
                kernel(i, j) = (1 / (2 * CV_PI * pow(sigma, 2))) * exp(-(pow((i - kernel_x0), 2) + pow((j - kernel_y0), 2)) / (2 * pow(sigma, 2)));
                sum += kernel(i, j);
            }
        }
        
        for (int i = 0; i < kernel_rows; i ++) {
            for (int j = 0; j < kernel_cols; j ++) {
                kernel(i, j) /= sum;
            }
        }
        return kernel;
    }
    cv::Mat gaussianBlur(const cv::Mat& src, const int kernel_size){
        Mat_<float> row_kernel = create_gaussian_kernel(kernel_size, 1);
        Mat_<float> row_filtered = applyConvolution<uchar>(src, row_kernel);
        Mat_<float> col_kernel = create_gaussian_kernel(1, kernel_size);
        return applyRawNormalization((applyConvolution<float>(row_filtered, col_kernel)), col_kernel);
    }

    cv::Mat_<int> computeIntegralImage(const cv::Mat_<uchar>& src) {
        int rows = src.rows;
        int cols = src.cols;
        
        cv::Mat_<int> integral = cv::Mat_<int>::zeros(rows + 1, cols + 1);

        for (int y = 1; y <= rows; y++) {
            for (int x = 1; x <= cols; x++) {
                integral(y, x) = src(y - 1, x - 1) 
                               + integral(y - 1, x) 
                               + integral(y, x - 1) 
                               - integral(y - 1, x - 1);
            }
        }
        return integral;
    }

    cv::Mat_<uchar> adaptiveThreshold(const cv::Mat_<uchar>& src, int windowSize, int C) {
        int rows = src.rows;
        int cols = src.cols;
        
        cv::Mat_<uchar> dst(rows, cols);
        
        cv::Mat_<int> integral = computeIntegralImage(src);
        
        int halfWin = windowSize / 2;

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                
                int y1 = std::max(0, y - halfWin);
                int y2 = std::min(rows - 1, y + halfWin);
                int x1 = std::max(0, x - halfWin);
                int x2 = std::min(cols - 1, x + halfWin);

                int D = integral(y2 + 1, x2 + 1);
                int B = integral(y1, x2 + 1);
                int C_val = integral(y2 + 1, x1);
                int A = integral(y1, x1);

                int sum = D - B - C_val + A;
                
                int count = (y2 - y1 + 1) * (x2 - x1 + 1);
                
                float local_mean = static_cast<float>(sum) / count;

                if (src(y, x) < (local_mean - C)) {
                    dst(y, x) = 255;  
                } else {
                    dst(y, x) = 0;    
                }
            }
        }
        return dst;
    }

    cv::Mat adaptiveThreshold(const cv::Mat& src, int windowSize, int c) {
        return adaptiveThreshold(cv::Mat_<uchar>(src), windowSize, c);
    }

   cv::Mat_<uchar> dilatation(const cv::Mat_<uchar>& src, int kWidth, int kHeight) {
        cv::Mat_<uchar> dst = cv::Mat_<uchar>::zeros(src.size()); 
        int halfW = kWidth / 2;
        int halfH = kHeight / 2;
        
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                if (src(i,j) == 255) { 
                    for (int u = -halfH; u <= halfH; u++) {
                        for (int v = -halfW; v <= halfW; v++) {
                            int r = i + u;
                            int c = j + v;
                            if (r >= 0 && r < src.rows && c >= 0 && c < src.cols) {
                                dst(r, c) = 255;
                            }
                        }
                    }
                }
            }
        }
        return dst;
    }

    cv::Mat_<uchar> erosion(const cv::Mat_<uchar>& src, int kWidth, int kHeight) {
        cv::Mat_<uchar> dst = cv::Mat_<uchar>::zeros(src.size()); 
        int halfW = kWidth / 2;
        int halfH = kHeight / 2;
        
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                if (src(i,j) == 255) {
                    bool keep = true;
                    for (int u = -halfH; u <= halfH; u++) {
                        for (int v = -halfW; v <= halfW; v++) {
                            int r = i + u;
                            int c = j + v;
                            if (r < 0 || r >= src.rows || c < 0 || c >= src.cols || src(r, c) == 0) {
                                keep = false;
                                break;
                            }
                        }
                        if (!keep) break; 
                    }
                    if (keep) dst(i, j) = 255;
                }
            }
        }
        return dst;
    }

    cv::Mat morphClose(const cv::Mat& src, int kernelSize) {
        cv::Mat_<uchar> dilated = dilatation(src, kernelSize, kernelSize);
        cv::Mat_<uchar> eroded = erosion(dilated, kernelSize, kernelSize);
        return eroded.clone();
    }
    cv::Mat morphClose(const cv::Mat& src, int kWidth, int kHeight) {
        cv::Mat_<uchar> dilated = dilatation(src, kWidth, kHeight);
        cv::Mat_<uchar> eroded = erosion(dilated, kWidth, kHeight);
        return eroded.clone();
    }

    std::vector<cv::Point> findLargestBlob(const cv::Mat& binaryImg) {
        int rows = binaryImg.rows;
        int cols = binaryImg.cols;
        
        cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8UC1);
        
        std::vector<cv::Point> largestBlob;
        size_t maxArea = 0;

        int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                
                if (binaryImg.at<uchar>(r, c) == 255 && visited.at<uchar>(r, c) == 0) {
                    
                    std::vector<cv::Point> currentBlob;
                    std::queue<cv::Point> q;
                    
                    q.push(cv::Point(c, r));
                    visited.at<uchar>(r, c) = 1;

                    while (!q.empty()) {
                        cv::Point p = q.front();
                        q.pop();
                        
                        currentBlob.push_back(p);

                        for (int i = 0; i < 8; ++i) {
                            int nr = p.y + dy[i];
                            int nc = p.x + dx[i];
                            
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                                if (binaryImg.at<uchar>(nr, nc) == 255 && visited.at<uchar>(nr, nc) == 0) {
                                    visited.at<uchar>(nr, nc) = 1;
                                    q.push(cv::Point(nc, nr));
                                }
                            }
                        }
                    }

                    if (currentBlob.size() > maxArea) {
                        maxArea = currentBlob.size();
                        largestBlob = std::move(currentBlob); 
                    }
                }
            }
        }
        
        return largestBlob;
    }
    std::array<cv::Point2f, 4> getGridCorners(const std::vector<cv::Point>& blobPixels) {
        cv::Point2f tl, tr, br, bl;
        
        int min_sum = INT_MAX, max_sum = -INT_MAX;
        int min_diff = INT_MAX, max_diff = -INT_MAX;

        for (const auto& pt : blobPixels) {
            int sum = pt.x + pt.y;
            int diff = pt.x - pt.y;

            // TL; x,y small
            if (sum < min_sum) { 
                min_sum = sum; 
                tl = pt; 
            }
            
            // BR; x, y large
            if (sum > max_sum) { 
                max_sum = sum; 
                br = pt; 
            }
            
            // TR ; x large, y small
            if (diff > max_diff) { 
                max_diff = diff; 
                tr = pt; 
            }
            
            // BL; x small y large
            if (diff < min_diff) { 
                min_diff = diff; 
                bl = pt; 
            }
        }

        return {tl, tr, br, bl};
    }
    static bool isDigitBlob(const Blob& b, int cellSize, const BlobFilterParams& p) {
        int cellArea = cellSize * cellSize;
        if (b.area < cellArea * p.minAreaFraction) {
            return false;  
        }
        if (b.area > cellArea * p.maxAreaFraction) {
            return false;
        }
        float aspect = (float)b.boundingBox.width / b.boundingBox.height;
        if (aspect > p.maxAspectRatio) {
            return false; 
        }
        if (aspect < p.minAspectRatio) {
            return false; 
        }
        return true;
    }
    std::vector<Blob> findBlobs(const cv::Mat& binary) {
        cv::Mat visited = cv::Mat::zeros(binary.size(), CV_8UC1);
        std::vector<Blob> blobs;

        int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

        for (int r = 0; r < binary.rows; ++r) {
            for (int c = 0; c < binary.cols; ++c) {
                if (binary.at<uchar>(r,c) == 0) continue;
                if (visited.at<uchar>(r,c)) continue;

                Blob blob;
                std::queue<cv::Point> q;
                q.push({c, r});
                visited.at<uchar>(r,c) = 1;

                int minX = c, maxX = c, minY = r, maxY = r;

                while (!q.empty()) {
                    cv::Point p = q.front(); q.pop();
                    blob.pixels.push_back(p);
                    minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
                    minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);

                    for (int d = 0; d < 8; ++d) {
                        cv::Point nb(p.x + dx[d], p.y + dy[d]);
                        if (nb.x < 0 || nb.y < 0 ||
                            nb.x >= binary.cols || nb.y >= binary.rows) continue;
                        if (visited.at<uchar>(nb.y, nb.x)) continue;
                        if (binary.at<uchar>(nb.y, nb.x) == 0) continue;  // skip background
                        visited.at<uchar>(nb.y, nb.x) = 1;
                        q.push(nb);
                    }
                }

                blob.area        = blob.pixels.size();
                blob.boundingBox = cv::Rect(minX, minY, maxX-minX+1, maxY-minY+1);
                blob.center      = cv::Point2f(
                                    (minX + maxX) / 2.f,
                                    (minY + maxY) / 2.f);
                blobs.push_back(blob);
            }
        }
        return blobs;
    }
    
    std::vector<CellDigit> assignToGrid(const std::vector<Blob>& digitBlobs, const cv::Mat& binary, int warpSize, int pad)
    {
        int cellSize = warpSize / 9;
        std::vector<CellDigit> result;
        
        // Initialize 81 blank cells
        std::vector<std::vector<CellDigit>> grid(9, std::vector<CellDigit>(9));
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                grid[r][c].row = r;
                grid[r][c].col = c;
                grid[r][c].cellImage = cv::Mat::zeros(cellSize, cellSize, CV_8UC1);
            }
        }

        // Place valid blobs on the grid
        for (const auto& b : digitBlobs) {
            int col = (int)(b.center.x / cellSize);
            int row = (int)(b.center.y / cellSize);

            col = std::clamp(col, 0, 8);
            row = std::clamp(row, 0, 8);

            int startX = std::max(0, b.boundingBox.x + pad);
            int startY = std::max(0, b.boundingBox.y + pad);
            int rectW  = std::min(binary.cols - startX, b.boundingBox.width - 2 * pad);
            int rectH  = std::min(binary.rows - startY, b.boundingBox.height - 2 * pad);

            cv::Rect padded(startX, startY, rectW, rectH);

            grid[row][col].cellImage = binary(padded).clone(); // Update the existing cell
        }

        // Flatten back into a 1D vector
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                result.push_back(grid[r][c]);
            }
        }
        return result;
    }

    std::vector<CellDigit> extractDigits(const cv::Mat& warpedBinary, int warpSize, const BlobFilterParams& params)
    {
        int cellSize = warpSize / 9;
        std::vector<Blob> allBlobs = findBlobs(warpedBinary);

        std::vector<Blob> digitBlobs;
        int noiseCount = 0;
        int largeCount = 0;
        int wideCount = 0;
        int thinCount = 0;

        for (const auto& b : allBlobs) {
            int cellArea = cellSize * cellSize;
            float aspect = (float)b.boundingBox.width / b.boundingBox.height;

            if (b.area < cellArea * params.minAreaFraction) noiseCount++;
            else if (b.area > cellArea * params.maxAreaFraction) largeCount++;
            else if (aspect > params.maxAspectRatio) wideCount++;
            else if (aspect < params.minAspectRatio) thinCount++;
            else digitBlobs.push_back(b);
        }
        
        std::cout << "[DEBUG] findBlobs found " << allBlobs.size() << " total blobs." << std::endl;
        std::cout << "[DEBUG] Rejected noise: " << noiseCount << ", too large: " << largeCount << ", too wide: " << wideCount << ", too thin: " << thinCount << std::endl;
        std::cout << "[DEBUG] isDigitBlob kept " << digitBlobs.size() << " digit blobs." << std::endl;
        
        return assignToGrid(digitBlobs, warpedBinary, warpSize, params.cropPadding);
    }
    cv::Mat normalise(const cv::Mat& cell, int size = 20) {
        cv::Mat resized;
        cv::resize(cell, resized, {size, size});
        return resized;
    }
    std::vector<float> getHorizontalProjection(const cv::Mat& cell) {
        cv::Mat norm = normalise(cell);
        const int N = norm.rows;
        std::vector<float> proj(N, 0.f);

        for (int r = 0; r < N; ++r)
            for (int c = 0; c < norm.cols; ++c)
                if (norm.at<uchar>(r, c) < 128)
                    proj[r] += 1.f;

        float maxVal = *std::max_element(proj.begin(), proj.end());
        if (maxVal > 0)
            for (float& v : proj) v /= maxVal;

        return proj;
    }

    std::vector<float> getVerticalProjection(const cv::Mat& cell) {
        cv::Mat norm = normalise(cell);
        const int N = norm.cols;
        std::vector<float> proj(N, 0.f);

        for (int r = 0; r < norm.rows; ++r)
            for (int c = 0; c < N; ++c)
                if (norm.at<uchar>(r, c) < 128)
                    proj[c] += 1.f;

        float maxVal = *std::max_element(proj.begin(), proj.end());
        if (maxVal > 0)
            for (float& v : proj) v /= maxVal;

        return proj;
    }
    float projectionDistance(const std::vector<float>& a, const std::vector<float>& b) {
        float dist = 0;
        for (int i = 0; i < (int)a.size(); ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return dist;
    }

}