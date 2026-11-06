#pragma once
#include "matrix.h"
#include <istream>
#include <ostream>
#include <cstdint>
#include <stdexcept>
#include <string>

// ============================================================
//  Binary serialization yardimcilari
//
//  Format (little-endian makinelerde yazilir, ayni mimaride okunur):
//    int32  satir, int32 sutun, ardindan satir*sutun adet double
//  Not: dosya, yazildigi makinenin bayt sirasini kullanir; farkli
//  mimariler arasi tasinabilirlik hedeflenmiyor.
// ============================================================

inline void write_i32(std::ostream& os, int32_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline int32_t read_i32(std::istream& is) {
    int32_t v = 0;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!is) throw std::runtime_error("Model dosyasi beklenenden kisa (bozuk olabilir)");
    return v;
}

inline void write_string(std::ostream& os, const std::string& s) {
    write_i32(os, (int32_t)s.size());
    os.write(s.data(), (std::streamsize)s.size());
}

inline std::string read_string(std::istream& is) {
    int32_t n = read_i32(is);
    if (n < 0 || n > 4096) throw std::runtime_error("Model dosyasi bozuk: gecersiz metin uzunlugu");
    std::string s(n, '\0');
    is.read(&s[0], n);
    if (!is) throw std::runtime_error("Model dosyasi beklenenden kisa (bozuk olabilir)");
    return s;
}

inline void write_matrix(std::ostream& os, const Matrix& m) {
    write_i32(os, m.rows);
    write_i32(os, m.cols);
    os.write(reinterpret_cast<const char*>(m.data.data()),
             (std::streamsize)(m.data.size() * sizeof(double)));
}

inline Matrix read_matrix(std::istream& is) {
    int32_t r = read_i32(is);
    int32_t c = read_i32(is);
    // Bozuk dosyada devasa bellek ayirmamak icin makul bir ust sinir
    if (r <= 0 || c <= 0 || (int64_t)r * c > 200000000LL)
        throw std::runtime_error("Model dosyasi bozuk: gecersiz matris boyutu");
    Matrix m(r, c);
    is.read(reinterpret_cast<char*>(m.data.data()),
            (std::streamsize)(m.data.size() * sizeof(double)));
    if (!is) throw std::runtime_error("Model dosyasi beklenenden kisa (bozuk olabilir)");
    return m;
}

// Dosyadan matris oku, boyutu hedefle ayni degilse hata ver, sonra kopyala
inline void read_into(std::istream& is, Matrix& target, const std::string& label) {
    Matrix m = read_matrix(is);
    if (m.rows != target.rows || m.cols != target.cols)
        throw std::runtime_error("Boyut uyusmazligi (" + label + "): dosyada "
            + std::to_string(m.rows) + "x" + std::to_string(m.cols) + ", agda "
            + std::to_string(target.rows) + "x" + std::to_string(target.cols));
    target = m;
}
