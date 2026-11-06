#include "matrix.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Dogruluk referansi: en basit, optimizasyonsuz ucl dongu (i,j,k)
static Matrix reference_matmul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < B.cols; j++) {
            double s = 0.0;
            for (int k = 0; k < A.cols; k++) s += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            C.data[i * C.cols + j] = s;
        }
    return C;
}

static double max_abs_diff(const Matrix& a, const Matrix& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.data.size(); i++) d = std::max(d, std::fabs(a.data[i] - b.data[i]));
    return d;
}

struct Shape { std::string name; int m, k, n; };

int main() {
    // MNIST agindaki gercek carpimlar (batch = 64) + genel bir kare matris
    std::vector<Shape> shapes = {
        {"L1 ileri   W(128x784)  * X(784x64)",   128, 784,  64},
        {"L1 dW      G(128x64)   * Xt(64x784)",  128,  64, 784},
        {"L1 dx      Wt(784x128) * G(128x64)",   784, 128,  64},
        {"L2 ileri   W(64x128)   * X(128x64)",    64, 128,  64},
        {"kare       A(512x512)  * B(512x512)",  512, 512, 512},
        // CNN (MNIST, batch 64): conv GEMM'leri kisa-genis ve uzun-dar sekillerdedir
        {"conv1 ileri  W(8x9)     * col(9x50176)",    8,    9, 50176},
        {"conv2 ileri  W(16x72)   * col(72x12544)",  16,   72, 12544},
        {"conv2 dW     col(72x12544) * dYt(12544x16)", 72, 12544,   16},
        {"conv2 dX     Wt(72x16)  * dY(16x12544)",   72,   16, 12544},
    };

    std::cout << "Carpim cekirdegi: " << Matrix::matmul_backend() << "\n\n";
    std::cout << "=== Dogruluk (optimize vs referans, tek sefer) ===\n";
    {
        // tuhaf boyutlar da dahil: blok/vektor kalanlarini yakalamak icin
        std::vector<Shape> odd = {{"", 1,1,1}, {"", 3,5,7}, {"", 17,33,19}, {"", 64,100,63},
                                  {"", 129,65,31}, {"", 5,257,130}};
        double worst = 0.0;
        for (auto& s : odd) {
            Matrix A = Matrix::random(s.m, s.k), B = Matrix::random(s.k, s.n);
            worst = std::max(worst, max_abs_diff(A * B, reference_matmul(A, B)));
        }
        std::cout << "  en buyuk mutlak fark: " << std::scientific << std::setprecision(2) << worst << "\n";
    }

    std::cout << "\n=== Hiz ===\n" << std::fixed;
    for (auto& s : shapes) {
        Matrix A = Matrix::random(s.m, s.k), B = Matrix::random(s.k, s.n);
        volatile double sink = 0;
        // isinma
        sink += (A * B).data[0];
        int reps = 0;
        auto t0 = std::chrono::steady_clock::now();
        double elapsed = 0.0;
        while (elapsed < 0.5) {
            sink += (A * B).data[0];
            reps++;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        double ms = 1000.0 * elapsed / reps;
        double gflops = 2.0 * s.m * s.k * s.n / (ms / 1000.0) / 1e9;
        std::cout << "  " << std::left << std::setw(40) << s.name << std::right
                  << std::setw(9) << std::setprecision(3) << ms << " ms  "
                  << std::setw(7) << std::setprecision(2) << gflops << " GFLOP/s\n";
    }
    return 0;
}
