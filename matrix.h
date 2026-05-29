#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>

class Matrix {
public:
    int rows, cols;
    std::vector<double> data;

    // Constructors
    Matrix(int rows, int cols, double val = 0.0);
    Matrix(int rows, int cols, std::vector<double> data);

    // Element access
    double& at(int r, int c);
    const double& at(int r, int c) const;

    // Core ops
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // matrix multiply
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

    // Matrix ops
    Matrix transpose() const;
    Matrix hadamard(const Matrix& other) const; // element-wise multiply
    Matrix apply(std::function<double(double)> func) const;

    // Static factories
    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);
    static Matrix random(int rows, int cols, double mean = 0.0, double stddev = 1.0);
    static Matrix from_vector(const std::vector<double>& v); // column vector

    // Reduction
    double sum() const;
    Matrix sum_rows() const;   // returns (1, cols)
    Matrix sum_cols() const;   // returns (rows, 1)

    // Utility
    void print(const std::string& name = "") const;
    Matrix clip(double min_val, double max_val) const;
};

// Free operator
Matrix operator*(double scalar, const Matrix& m);
