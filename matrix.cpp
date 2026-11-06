#include "matrix.h"
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <string>

// ---------- Constructors ----------

Matrix::Matrix(int rows, int cols, double val)
    : rows(rows), cols(cols), data(rows * cols, val) {}

Matrix::Matrix(int rows, int cols, std::vector<double> d)
    : rows(rows), cols(cols), data(std::move(d)) {
    assert((int)data.size() == rows * cols);
}

// ---------- Element access ----------

double& Matrix::at(int r, int c) {
    return data[r * cols + c];
}

const double& Matrix::at(int r, int c) const {
    return data[r * cols + c];
}

// ---------- Arithmetic ----------

Matrix Matrix::operator+(const Matrix& o) const {
    assert(rows == o.rows && cols == o.cols);
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        result.data[i] = data[i] + o.data[i];
    return result;
}

Matrix Matrix::operator-(const Matrix& o) const {
    assert(rows == o.rows && cols == o.cols);
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        result.data[i] = data[i] - o.data[i];
    return result;
}

// ============================================================
//  Matris carpimi: C (m x n) = A (m x k) * B (k x n)   (satir-major)
//
//  Optimizasyonlar:
//   1) Register bloklama: C'nin 4xNR'lik bir parcasi SIMD register'larinda
//      tutulur ve k boyunca biriktirilir; C'ye bellek erisimi k kez degil
//      1 kez olur.
//   2) Acik SIMD: AVX2 (4 double) / SSE2 (2 double) intrinsic'leri.
//      Derleyicinin otomatik vektorlemesi bayraga ve surume gore cok
//      oynak oldugu icin (olculdu) elle yazildi. CPU calisma aninda secilir.
//   3) k bloklama (KC): B'nin kullanilan dilimi onbellekte kalir.
//   4) OpenMP (derlemede -fopenmp / /openmp varsa): C'nin satir bloklari
//      farkli thread'lerde hesaplanir; her C elemanini tek thread yazar,
//      sonuc thread sayisindan bagimsizdir.
//
//  Her C[i][j] icin toplama sirasi k = 0,1,2,... KORUNUR ve carpma/toplama
//  ayri yapilir (FMA yok) -> uc yolun (AVX2 / SSE2 / skaler) sonucu bit-bit
//  AYNIDIR. (Derleyiciye global -mfma/-march=native verirsen derleyici
//  carpma+toplamayi FMA'ya birlestirebilir; sonuc 1e-14 mertebesinde degisir.)
//
//  Kapatmak icin: -DNN_NO_SIMD
// ============================================================

#if !defined(NN_NO_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    #define NN_X86 1
    #include <immintrin.h>
#else
    #define NN_X86 0
#endif

// AVX2 cekirdegi: GCC/Clang'da fonksiyon basina hedef, MSVC'de /arch:AVX2 gerekir
#if NN_X86 && (defined(__GNUC__) || defined(__clang__))
    #define NN_HAVE_AVX2_KERNEL 1
    #define NN_TARGET_AVX2 __attribute__((target("avx2")))
#elif NN_X86 && defined(_MSC_VER) && defined(__AVX2__)
    #define NN_HAVE_AVX2_KERNEL 1
    #define NN_TARGET_AVX2
#else
    #define NN_HAVE_AVX2_KERNEL 0
    #define NN_TARGET_AVX2
#endif

namespace {

constexpr int KC = 256;   // k blok boyu

// ---------- Skaler yedek cekirdek (her platformda calisir) ----------
struct KernelScalar {
    static constexpr int MR = 4, NR = 8;
    static const char* name() { return "scalar"; }
    static void run(const double* __restrict A, int lda,
                    const double* __restrict B, int ldb,
                    double* __restrict C, int ldc, int k) {
        double c0[NR], c1[NR], c2[NR], c3[NR];
        for (int c = 0; c < NR; c++) {
            c0[c] = C[c];           c1[c] = C[ldc + c];
            c2[c] = C[2 * ldc + c]; c3[c] = C[3 * ldc + c];
        }
        for (int p = 0; p < k; p++) {
            const double* b = B + (long)p * ldb;
            const double x0 = A[p], x1 = A[lda + p], x2 = A[2 * lda + p], x3 = A[3 * lda + p];
            for (int c = 0; c < NR; c++) {
                const double bc = b[c];
                c0[c] += x0 * bc; c1[c] += x1 * bc; c2[c] += x2 * bc; c3[c] += x3 * bc;
            }
        }
        for (int c = 0; c < NR; c++) {
            C[c] = c0[c];           C[ldc + c] = c1[c];
            C[2 * ldc + c] = c2[c]; C[3 * ldc + c] = c3[c];
        }
    }
};

#if NN_X86
// ---------- SSE2 cekirdegi: 4x4, 8 adet xmm akumulator ----------
struct KernelSSE2 {
    static constexpr int MR = 4, NR = 4;
    static const char* name() { return "SSE2"; }
    static void run(const double* __restrict A, int lda,
                    const double* __restrict B, int ldb,
                    double* __restrict C, int ldc, int k) {
        __m128d c00 = _mm_loadu_pd(C),           c01 = _mm_loadu_pd(C + 2);
        __m128d c10 = _mm_loadu_pd(C + ldc),     c11 = _mm_loadu_pd(C + ldc + 2);
        __m128d c20 = _mm_loadu_pd(C + 2 * ldc), c21 = _mm_loadu_pd(C + 2 * ldc + 2);
        __m128d c30 = _mm_loadu_pd(C + 3 * ldc), c31 = _mm_loadu_pd(C + 3 * ldc + 2);
        const double *a0 = A, *a1 = A + lda, *a2 = A + 2 * lda, *a3 = A + 3 * lda;
        for (int p = 0; p < k; p++) {
            const double* b = B + (long)p * ldb;
            const __m128d b0 = _mm_loadu_pd(b), b1 = _mm_loadu_pd(b + 2);
            __m128d a;
            a = _mm_set1_pd(a0[p]); c00 = _mm_add_pd(c00, _mm_mul_pd(a, b0)); c01 = _mm_add_pd(c01, _mm_mul_pd(a, b1));
            a = _mm_set1_pd(a1[p]); c10 = _mm_add_pd(c10, _mm_mul_pd(a, b0)); c11 = _mm_add_pd(c11, _mm_mul_pd(a, b1));
            a = _mm_set1_pd(a2[p]); c20 = _mm_add_pd(c20, _mm_mul_pd(a, b0)); c21 = _mm_add_pd(c21, _mm_mul_pd(a, b1));
            a = _mm_set1_pd(a3[p]); c30 = _mm_add_pd(c30, _mm_mul_pd(a, b0)); c31 = _mm_add_pd(c31, _mm_mul_pd(a, b1));
        }
        _mm_storeu_pd(C, c00);           _mm_storeu_pd(C + 2, c01);
        _mm_storeu_pd(C + ldc, c10);     _mm_storeu_pd(C + ldc + 2, c11);
        _mm_storeu_pd(C + 2 * ldc, c20); _mm_storeu_pd(C + 2 * ldc + 2, c21);
        _mm_storeu_pd(C + 3 * ldc, c30); _mm_storeu_pd(C + 3 * ldc + 2, c31);
    }
};
#endif

#if NN_HAVE_AVX2_KERNEL
// ---------- AVX2 cekirdegi: 4x8, 8 adet ymm akumulator ----------
struct KernelAVX2 {
    static constexpr int MR = 4, NR = 8;
    static const char* name() { return "AVX2"; }
    NN_TARGET_AVX2
    static void run(const double* __restrict A, int lda,
                    const double* __restrict B, int ldb,
                    double* __restrict C, int ldc, int k) {
        __m256d c00 = _mm256_loadu_pd(C),           c01 = _mm256_loadu_pd(C + 4);
        __m256d c10 = _mm256_loadu_pd(C + ldc),     c11 = _mm256_loadu_pd(C + ldc + 4);
        __m256d c20 = _mm256_loadu_pd(C + 2 * ldc), c21 = _mm256_loadu_pd(C + 2 * ldc + 4);
        __m256d c30 = _mm256_loadu_pd(C + 3 * ldc), c31 = _mm256_loadu_pd(C + 3 * ldc + 4);
        const double *a0 = A, *a1 = A + lda, *a2 = A + 2 * lda, *a3 = A + 3 * lda;
        for (int p = 0; p < k; p++) {
            const double* b = B + (long)p * ldb;
            const __m256d b0 = _mm256_loadu_pd(b), b1 = _mm256_loadu_pd(b + 4);
            __m256d a;
            a = _mm256_broadcast_sd(a0 + p); c00 = _mm256_add_pd(c00, _mm256_mul_pd(a, b0)); c01 = _mm256_add_pd(c01, _mm256_mul_pd(a, b1));
            a = _mm256_broadcast_sd(a1 + p); c10 = _mm256_add_pd(c10, _mm256_mul_pd(a, b0)); c11 = _mm256_add_pd(c11, _mm256_mul_pd(a, b1));
            a = _mm256_broadcast_sd(a2 + p); c20 = _mm256_add_pd(c20, _mm256_mul_pd(a, b0)); c21 = _mm256_add_pd(c21, _mm256_mul_pd(a, b1));
            a = _mm256_broadcast_sd(a3 + p); c30 = _mm256_add_pd(c30, _mm256_mul_pd(a, b0)); c31 = _mm256_add_pd(c31, _mm256_mul_pd(a, b1));
        }
        _mm256_storeu_pd(C, c00);           _mm256_storeu_pd(C + 4, c01);
        _mm256_storeu_pd(C + ldc, c10);     _mm256_storeu_pd(C + ldc + 4, c11);
        _mm256_storeu_pd(C + 2 * ldc, c20); _mm256_storeu_pd(C + 2 * ldc + 4, c21);
        _mm256_storeu_pd(C + 3 * ldc, c30); _mm256_storeu_pd(C + 3 * ldc + 4, c31);
    }
};
#endif

// C[0:m, 0:n] += A[0:m, 0:k] * B[0:k, 0:n]   (K: secilen mikro-cekirdek)
template <class K>
void gemm_block(const double* __restrict A, int lda,
                const double* __restrict B, int ldb,
                double* __restrict C, int ldc,
                int m, int k, int n) {
    constexpr int MR = K::MR, NR = K::NR;
    const int mm = m - m % MR;   // MR'nin kati olan satir sayisi
    const int nn = n - n % NR;   // NR'nin kati olan sutun sayisi

    for (int kb = 0; kb < k; kb += KC) {
        const int kl = std::min(KC, k - kb);
        const double* Ab = A + kb;
        const double* Bb = B + (long)kb * ldb;

        // ana bolge: MR x NR'lik tam bloklar
        for (int j = 0; j < nn; j += NR)
            for (int i = 0; i < mm; i += MR)
                K::run(Ab + (long)i * lda, lda, Bb + j, ldb,
                       C + (long)i * ldc + j, ldc, kl);

        // kalan sutunlar (n % NR) - ilk mm satir
        if (nn < n)
            for (int i = 0; i < mm; i++)
                for (int p = 0; p < kl; p++) {
                    const double a = Ab[(long)i * lda + p];
                    const double* b = Bb + (long)p * ldb;
                    double* c = C + (long)i * ldc;
                    for (int j = nn; j < n; j++) c[j] += a * b[j];
                }

        // kalan satirlar (m % MR) - tum sutunlar
        for (int i = mm; i < m; i++)
            for (int p = 0; p < kl; p++) {
                const double a = Ab[(long)i * lda + p];
                const double* b = Bb + (long)p * ldb;
                double* c = C + (long)i * ldc;
                for (int j = 0; j < n; j++) c[j] += a * b[j];
            }
    }
}

using BlockFn = void (*)(const double*, int, const double*, int, double*, int, int, int, int);
struct Backend { BlockFn fn; const char* name; };

#if NN_HAVE_AVX2_KERNEL
bool cpu_has_avx2() {
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    #else
        return true;   // MSVC: /arch:AVX2 ile derlendiyse derleme zamaninda garanti
    #endif
}
#endif

// CPU'ya gore bir kez secilir.
// Deney/test icin zorlama: NN_MATMUL=scalar | sse2 | avx2 (ortam degiskeni)
Backend pick_backend() {
    const char* force = std::getenv("NN_MATMUL");
    if (force) {
        const std::string f(force);
        if (f == "scalar") return { &gemm_block<KernelScalar>, KernelScalar::name() };
#if NN_X86
        if (f == "sse2")   return { &gemm_block<KernelSSE2>, KernelSSE2::name() };
#endif
#if NN_HAVE_AVX2_KERNEL
        if (f == "avx2" && cpu_has_avx2()) return { &gemm_block<KernelAVX2>, KernelAVX2::name() };
#endif
    }
#if NN_HAVE_AVX2_KERNEL
    if (cpu_has_avx2()) return { &gemm_block<KernelAVX2>, KernelAVX2::name() };
#endif
#if NN_X86
    return { &gemm_block<KernelSSE2>, KernelSSE2::name() };
#else
    return { &gemm_block<KernelScalar>, KernelScalar::name() };
#endif
}

const Backend& backend() {
    static const Backend b = pick_backend();
    return b;
}

// C (zaten sifirlanmis) += A * B ; buyuk carpimlarda satir bloklari paralel
void gemm(const double* A, const double* B, double* C, int m, int k, int n) {
    if (m == 0 || k == 0 || n == 0) return;
    const BlockFn fn = backend().fn;
#ifdef _OPENMP
    const long work = (long)m * k * n;
    if (work >= 2000000L) {                         // kucuk carpimda thread acmaya degmez
        const int MC = 16;                          // satir gorevi: 16 satir
        const int NC = 512;                         // sutun gorevi: 512 sutun (NR'nin kati)
        const int rtasks = (m + MC - 1) / MC;
        const int ctasks = (n + NC - 1) / NC;
        if (n >= 2 * NC && ctasks > rtasks) {
            // kisa-genis sekil (ornegin conv: m = cikis kanali sayisi = 8..72,
            // n = pozisyon*batch ~ 10^4): satir bolmek az gorev uretir, SUTUNLARA bol.
            // Her C elemanini yine tek thread yazar ve k siralamasi degismez -> sonuc ayni.
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < ctasks; t++) {
                const int j0 = t * NC;
                const int nj = std::min(NC, n - j0);
                fn(A, k, B + j0, n, C + j0, n, m, k, nj);
            }
            return;
        }
        if (m >= 8) {
            // yeterince satir var: satir bloklarina bol
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < rtasks; t++) {
                const int i0 = t * MC;
                const int mi = std::min(MC, m - i0);
                fn(A + (long)i0 * k, k, B, n, C + (long)i0 * n, n, mi, k, n);
            }
            return;
        }
    }
#endif
    fn(A, k, B, n, C, n, m, k, n);
}

} // namespace

const char* Matrix::matmul_backend() { return backend().name; }

// Matrix multiplication: (m x n) * (n x p) -> (m x p)
Matrix Matrix::operator*(const Matrix& o) const {
    assert(cols == o.rows);
    Matrix result(rows, o.cols, 0.0);
    gemm(data.data(), o.data.data(), result.data.data(), rows, cols, o.cols);
    return result;
}

Matrix Matrix::operator*(double s) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        result.data[i] = data[i] * s;
    return result;
}

Matrix Matrix::operator/(double s) const {
    return (*this) * (1.0 / s);
}

Matrix& Matrix::operator+=(const Matrix& o) {
    assert(rows == o.rows && cols == o.cols);
    for (int i = 0; i < rows * cols; i++)
        data[i] += o.data[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& o) {
    assert(rows == o.rows && cols == o.cols);
    for (int i = 0; i < rows * cols; i++)
        data[i] -= o.data[i];
    return *this;
}

Matrix operator*(double s, const Matrix& m) { return m * s; }

// ---------- Matrix ops ----------

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.at(j, i) = at(i, j);
    return result;
}

Matrix Matrix::hadamard(const Matrix& o) const {
    assert(rows == o.rows && cols == o.cols);
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        result.data[i] = data[i] * o.data[i];
    return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
        result.data[i] = func(data[i]);
    return result;
}

// ---------- Static factories ----------

Matrix Matrix::zeros(int r, int c) { return Matrix(r, c, 0.0); }
Matrix Matrix::ones(int r, int c)  { return Matrix(r, c, 1.0); }

static std::mt19937& global_rng() {
    static std::mt19937 rng(42);
    return rng;
}

void Matrix::set_seed(unsigned seed) { global_rng().seed(seed); }

Matrix Matrix::random(int r, int c, double mean, double stddev) {
    std::mt19937& rng = global_rng();
    std::normal_distribution<double> dist(mean, stddev);
    Matrix result(r, c);
    for (auto& v : result.data) v = dist(rng);
    return result;
}

Matrix Matrix::from_vector(const std::vector<double>& v) {
    Matrix result((int)v.size(), 1);
    result.data = v;
    return result;
}

// ---------- Reductions ----------

double Matrix::sum() const {
    double s = 0.0;
    for (auto v : data) s += v;
    return s;
}

Matrix Matrix::sum_rows() const {   // sum across rows -> (1, cols)
    Matrix result = Matrix::zeros(1, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.at(0, j) += at(i, j);
    return result;
}

Matrix Matrix::sum_cols() const {   // sum across cols -> (rows, 1)
    Matrix result = Matrix::zeros(rows, 1);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.at(i, 0) += at(i, j);
    return result;
}

// ---------- Utility ----------

Matrix Matrix::clip(double lo, double hi) const {
    return apply([lo, hi](double x){ return std::clamp(x, lo, hi); });
}

void Matrix::print(const std::string& name) const {
    if (!name.empty()) std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        std::cout << "  [";
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(9) << std::setprecision(4) << std::fixed << at(i, j);
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}
