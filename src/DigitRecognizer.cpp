#include "../include/DigitRecognizer.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

static constexpr int FG_THRESH = 128;

cv::Mat DigitRecognizer::normalise(const cv::Mat& cell) {
    if (cell.empty()) return {};

    int minX = cell.cols, minY = cell.rows, maxX = -1, maxY = -1;
    int fgCount = 0;
    for (int y = 0; y < cell.rows; ++y) {
        for (int x = 0; x < cell.cols; ++x) {
            if (cell.at<uchar>(y, x) > FG_THRESH) {
                ++fgCount;
                minX = std::min(minX, x); maxX = std::max(maxX, x);
                minY = std::min(minY, y); maxY = std::max(maxY, y);
            }
        }
    }
    if (fgCount < 8 || maxX < minX) return {}; //there is too little ink

    cv::Mat crop = cell(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1));

    
    const int inner = S - 2 * MARGIN;
    double sc = std::min(inner / (double)crop.rows, inner / (double)crop.cols);
    int nw = std::max(1, (int)std::lround(crop.cols * sc));
    int nh = std::max(1, (int)std::lround(crop.rows * sc));

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    cv::Mat bin;
    cv::threshold(resized, bin, 96, 255, cv::THRESH_BINARY);

    double sumX = 0, sumY = 0; int cnt = 0;
    for (int y = 0; y < bin.rows; ++y)
        for (int x = 0; x < bin.cols; ++x)
            if (bin.at<uchar>(y, x) > FG_THRESH) { sumX += x; sumY += y; ++cnt; }
    if (cnt == 0) return {};

    double cx = sumX / cnt, cy = sumY / cnt;
    int ox = (int)std::lround(S / 2.0 - cx);
    int oy = (int)std::lround(S / 2.0 - cy);
    ox = std::clamp(ox, 0, S - nw);
    oy = std::clamp(oy, 0, S - nh);

    cv::Mat canvas = cv::Mat::zeros(S, S, CV_8UC1);
    bin.copyTo(canvas(cv::Rect(ox, oy, nw, nh)));
    return canvas;
}

DigitFeatures DigitRecognizer::extractFeatures(const cv::Mat& norm) {
    DigitFeatures f;
    if (norm.empty()) return f;  

    // comparatie pixel cu pixel 
    f.pixels.reserve(S * S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x)
            f.pixels.push_back(norm.at<uchar>(y, x) > FG_THRESH ? 1.f : 0.f);

    //denistate pe zone (pentru 1 si 7 sau 3 si 8)
    const int zs = S / Z;
    f.zoning.assign(Z * Z, 0.f);
    for (int i = 0; i < Z; ++i) {
        for (int j = 0; j < Z; ++j) {
            int sum = 0;
            for (int y = i * zs; y < (i + 1) * zs; ++y)
                for (int x = j * zs; x < (j + 1) * zs; ++x)
                    if (norm.at<uchar>(y, x) > FG_THRESH) ++sum;
            f.zoning[i * Z + j] = (float)sum / (zs * zs);
        }
    }

    // proiectii
    f.h_proj.assign(S, 0.f);
    f.v_proj.assign(S, 0.f);
    for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
            if (norm.at<uchar>(y, x) > FG_THRESH) {
                f.h_proj[y] += 1.f;
                f.v_proj[x] += 1.f;
            }
        }
    }
    for (float& v : f.h_proj) v /= S;
    for (float& v : f.v_proj) v /= S;

    f.valid = true;
    return f;
}

float DigitRecognizer::calculateDistance(const DigitFeatures& a,
                                         const DigitFeatures& b) {
    auto meanSq = [](const std::vector<float>& u, const std::vector<float>& v) {
        float d = 0.f;
        for (size_t i = 0; i < u.size(); ++i) { float t = u[i] - v[i]; d += t * t; }
        return u.empty() ? 0.f : d / u.size();
    };
    auto sumSq = [](const std::vector<float>& u, const std::vector<float>& v) {
        float d = 0.f;
        for (size_t i = 0; i < u.size(); ++i) { float t = u[i] - v[i]; d += t * t; }
        return d;
    };
    return 3.0f * meanSq(a.pixels, b.pixels) // cea mai imp e comp pixel cu pixel
         + 1.0f * sumSq(a.zoning, b.zoning)
         + 0.3f * sumSq(a.h_proj, b.h_proj)
         + 0.3f * sumSq(a.v_proj, b.v_proj);
}

void DigitRecognizer::addSample(int digit, const cv::Mat& image) {
    DigitFeatures f = extractFeatures(normalise(image));
    if (f.valid) references.emplace_back(digit, std::move(f));
}

static int loadDir(const std::string& dir,
                   const std::function<void(int, const cv::Mat&)>& add) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return 0;
    int loaded = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        std::string name = e.path().filename().string();
        if (name.rfind("digit_", 0) != 0) continue;
        int label = name[6] - '0';
        if (label < 1 || label > 9) continue;
        cv::Mat img = cv::imread(e.path().string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        add(label, img);
        ++loaded;
    }
    return loaded;
}

DigitRecognizer::DigitRecognizer() {
    auto add = [this](int d, const cv::Mat& m) { addSample(d, m); };

    int harvested = loadDir("templates_knn", add);
    int legacy = loadDir("templates", add);

    std::cout << "[DigitRecognizer] loaded " << harvested
              << " harvested + " << legacy << " legacy samples ("
              << references.size() << " total)\n";

    if (references.empty())
        throw std::runtime_error(
            "DigitRecognizer: no reference samples found. Expected digit images "
            "in templates_knn/ (run with --harvest on a solvable grid first).");
}

int DigitRecognizer::recognizeDigit(const cv::Mat& cell) {
    DigitFeatures query = extractFeatures(normalise(cell));
    if (!query.valid) return 0; 

    // Distance to every reference sample.
    std::vector<std::pair<float, int>> dists;  // (distance, digit)
    dists.reserve(references.size());
    for (const auto& [digit, feat] : references)
        dists.emplace_back(calculateDistance(query, feat), digit);

    // Weighted vote over the K nearest neighbours (weight = 1 / (dist + eps)).
    int k = std::min<int>(K, (int)dists.size());
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

    std::array<float, 10> votes{};
    for (int i = 0; i < k; ++i)
        votes[dists[i].second] += 1.0f / (dists[i].first + 1e-6f);

    int best = 0;
    float bestVote = -1.f;
    for (int d = 1; d <= 9; ++d)
        if (votes[d] > bestVote) { bestVote = votes[d]; best = d; }
    return best;
}
