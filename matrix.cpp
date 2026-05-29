#include "matrix.h"
#include <iomanip>
#include <algorithm>

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

// Matrix multiplication: (m x n) * (n x p) -> (m x p)
Matrix Matrix::operator*(const Matrix& o) const {
    assert(cols == o.rows);
    Matrix result(rows, o.cols, 0.0);
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < cols; k++)
            for (int j = 0; j < o.cols; j++)
                result.at(i, j) += at(i, k) * o.at(k, j);
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

Matrix Matrix::random(int r, int c, double mean, double stddev) {
    static std::mt19937 rng(42);
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
