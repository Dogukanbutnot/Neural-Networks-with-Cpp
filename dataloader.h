#pragma once
#include "matrix.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <utility>
#include <stdexcept>

// ============================================================
//  DataLoader
//
//  Veri düzeni: X (features x N), Y (outputs x N)
//  Her SÜTUN bir örnek — ağımızın beklediği düzenle aynı.
//
//  Kullanım:
//    DataLoader loader(X, Y, /*batch_size=*/32);
//    for (int epoch = 0; epoch < E; epoch++) {
//        loader.on_epoch_start();               // shuffle
//        for (int b = 0; b < loader.num_batches(); b++) {
//            auto [xb, yb] = loader.get_batch(b);
//            ...
//        }
//    }
//
//  Shuffle neden önemli?
//  Veri sıralıysa (ör. önce tüm 0'lar sonra tüm 1'ler) her batch
//  tek sınıftan oluşur ve gradyanlar sürekli aynı yöne çeker.
//  Her epoch'ta yeniden karıştırmak bunu engeller.
// ============================================================
class DataLoader {
    Matrix X, Y;
    int  n_samples;
    int  batch_size;
    bool shuffle;
    bool drop_last;
    std::vector<int> indices;
    std::mt19937 rng;

public:
    DataLoader(const Matrix& X, const Matrix& Y,
               int batch_size,
               bool shuffle   = true,
               bool drop_last = false,
               unsigned seed  = 42)
        : X(X), Y(Y),
          n_samples(X.cols),
          batch_size(batch_size),
          shuffle(shuffle),
          drop_last(drop_last),
          indices(X.cols),
          rng(seed)
    {
        if (X.cols != Y.cols)
            throw std::invalid_argument("DataLoader: X ve Y ayni sayida orneğe sahip olmali");
        if (batch_size <= 0)
            throw std::invalid_argument("DataLoader: batch_size > 0 olmali");
        std::iota(indices.begin(), indices.end(), 0);
    }

    int num_samples() const { return n_samples; }

    int num_batches() const {
        if (drop_last) return n_samples / batch_size;
        return (n_samples + batch_size - 1) / batch_size;   // tavan bölme
    }

    // Her epoch başında çağır — indeksleri karıştırır
    void on_epoch_start() {
        if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);
    }

    // b. batch'i döndür: (xb, yb)
    // Son batch eksik olabilir (drop_last=false ise)
    std::pair<Matrix, Matrix> get_batch(int b) const {
        if (b < 0 || b >= num_batches())
            throw std::out_of_range("DataLoader: gecersiz batch indeksi");

        int start = b * batch_size;
        int end   = std::min(start + batch_size, n_samples);
        int count = end - start;

        Matrix xb(X.rows, count);
        Matrix yb(Y.rows, count);

        // Satir satir kopyala: hem kaynak satiri hem hedef satiri ardisik
        // okunur/yazilir (sutun sutun kopyalamak her adimda cok uzaga atlar
        // ve onbellegi bosa harcar).
        const int* idx = indices.data() + start;
        for (int i = 0; i < X.rows; i++) {
            const double* src = X.data.data() + (size_t)i * X.cols;
            double*       dst = xb.data.data() + (size_t)i * count;
            for (int j = 0; j < count; j++) dst[j] = src[idx[j]];
        }
        for (int i = 0; i < Y.rows; i++) {
            const double* src = Y.data.data() + (size_t)i * Y.cols;
            double*       dst = yb.data.data() + (size_t)i * count;
            for (int j = 0; j < count; j++) dst[j] = src[idx[j]];
        }
        return {xb, yb};
    }
};

// Sınıf etiketlerinden (0..C-1) one-hot matris üret: (classes x N)
inline Matrix one_hot(const std::vector<int>& labels, int num_classes) {
    Matrix Y(num_classes, (int)labels.size(), 0.0);
    for (int j = 0; j < (int)labels.size(); j++) {
        if (labels[j] < 0 || labels[j] >= num_classes)
            throw std::out_of_range("one_hot: etiket aralik disi");
        Y.at(labels[j], j) = 1.0;
    }
    return Y;
}

// X'in [start, end) sutun araligini kopyala (egitim/dogrulama ayirmak icin)
inline Matrix slice_columns(const Matrix& X, int start, int end) {
    if (start < 0 || end > X.cols || start >= end)
        throw std::out_of_range("slice_columns: gecersiz aralik");
    Matrix out(X.rows, end - start);
    const int w = end - start;
    for (int i = 0; i < X.rows; i++)       // her satirin [start,end) dilimi ardisik
        std::copy(X.data.begin() + (size_t)i * X.cols + start,
                  X.data.begin() + (size_t)i * X.cols + end,
                  out.data.begin() + (size_t)i * w);
    return out;
}

inline std::vector<int> slice_labels(const std::vector<int>& v, int start, int end) {
    if (start < 0 || end > (int)v.size() || start >= end)
        throw std::out_of_range("slice_labels: gecersiz aralik");
    return std::vector<int>(v.begin() + start, v.begin() + end);
}
