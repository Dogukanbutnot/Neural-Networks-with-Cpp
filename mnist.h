#pragma once
#include "matrix.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
//  MNIST IDX okuyucu
//
//  IDX formati (tum sayilar BIG-ENDIAN int32):
//    goruntuler: magic=2051 | adet | satir(28) | sutun(28) | adet*784 bayt
//    etiketler : magic=2049 | adet | adet bayt
//
//  Dikkat: .gz dosyalarini once ACMAN gerekir (zlib kullanmiyoruz):
//    Linux/macOS: gunzip *.gz        Windows: 7-Zip ile "Extract here"
//
//  Cikti duzeni ag ile ayni: X (784 x N), her SUTUN bir goruntu,
//  piksel degerleri 0..255 -> 0..1 araligina olceklenir.
// ============================================================

namespace mnist_detail {
    inline uint32_t read_be32(std::istream& is) {
        unsigned char b[4];
        is.read(reinterpret_cast<char*>(b), 4);
        if (!is) throw std::runtime_error("MNIST dosyasi beklenenden kisa");
        return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16)
             | (uint32_t(b[2]) << 8)  |  uint32_t(b[3]);
    }
}

// max_samples > 0 ise sadece ilk max_samples ornegi yukler (hizli deneme icin)
inline Matrix load_mnist_images(const std::string& path, int max_samples = 0) {
    using namespace mnist_detail;
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("MNIST dosyasi acilamadi: " + path);

    uint32_t magic = read_be32(is);
    if (magic != 2051) {
        if ((magic >> 16) == 0x1f8b)   // gzip imzasi (0x1f 0x8b)
            throw std::runtime_error("Bu bir .gz dosyasi, once acmalisin: " + path);
        throw std::runtime_error("Gecersiz goruntu dosyasi (magic=" + std::to_string(magic) + "): " + path);
    }
    uint32_t n    = read_be32(is);
    uint32_t rows = read_be32(is);
    uint32_t cols = read_be32(is);
    if (rows != 28 || cols != 28)
        throw std::runtime_error("Beklenmeyen goruntu boyutu (28x28 olmali): " + path);
    if (max_samples > 0 && (uint32_t)max_samples < n) n = max_samples;

    const int pixels = (int)(rows * cols);
    Matrix X(pixels, (int)n);
    std::vector<unsigned char> buf(pixels);
    for (uint32_t j = 0; j < n; j++) {
        is.read(reinterpret_cast<char*>(buf.data()), pixels);
        if (!is) throw std::runtime_error("MNIST goruntu dosyasi beklenenden kisa: " + path);
        for (int i = 0; i < pixels; i++)
            X.at(i, (int)j) = buf[i] / 255.0;
    }
    return X;
}

inline std::vector<int> load_mnist_labels(const std::string& path, int max_samples = 0) {
    using namespace mnist_detail;
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("MNIST dosyasi acilamadi: " + path);

    uint32_t magic = read_be32(is);
    if (magic != 2049) {
        if ((magic >> 16) == 0x1f8b)
            throw std::runtime_error("Bu bir .gz dosyasi, once acmalisin: " + path);
        throw std::runtime_error("Gecersiz etiket dosyasi (magic=" + std::to_string(magic) + "): " + path);
    }
    uint32_t n = read_be32(is);
    if (max_samples > 0 && (uint32_t)max_samples < n) n = max_samples;

    std::vector<unsigned char> buf(n);
    is.read(reinterpret_cast<char*>(buf.data()), n);
    if (!is) throw std::runtime_error("MNIST etiket dosyasi beklenenden kisa: " + path);

    std::vector<int> labels(n);
    for (uint32_t j = 0; j < n; j++) {
        if (buf[j] > 9) throw std::runtime_error("Gecersiz etiket degeri: " + path);
        labels[j] = buf[j];
    }
    return labels;
}
