#pragma once
#include "matrix.h"
#include <unordered_map>
#include <string>
#include <cmath>

// ============================================================
//  Optimizer abstract base
// ============================================================
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Matrix& param, const Matrix& grad,
                      const std::string& param_id) = 0;
    virtual std::string name() const = 0;
};

// ============================================================
//  SGD  (karşılaştırma için)
// ============================================================
class SGD : public Optimizer {
    double lr;
public:
    explicit SGD(double lr = 0.01) : lr(lr) {}

    void step(Matrix& param, const Matrix& grad,
              const std::string& /*id*/) override {
        param -= grad * lr;
    }

    std::string name() const override {
        return "SGD(lr=" + std::to_string(lr) + ")";
    }
};

// ============================================================
//  Adam  —  Kingma & Ba (2014)
//
//  m_t = β1*m_{t-1} + (1-β1)*g           ← 1. moment (ortalama)
//  v_t = β2*v_{t-1} + (1-β2)*g²          ← 2. moment (varyans)
//  m̂   = m_t / (1-β1^t)                  ← bias düzeltmesi
//  v̂   = v_t / (1-β2^t)                  ← bias düzeltmesi
//  θ   = θ - α * m̂ / (√v̂ + ε)
//
//  Varsayılan hiperparametreler (orijinal paper):
//    α  = 0.001
//    β1 = 0.9
//    β2 = 0.999
//    ε  = 1e-8
// ============================================================
class Adam : public Optimizer {
    double alpha;    // öğrenme oranı
    double beta1;    // 1. moment bozunma katsayısı
    double beta2;    // 2. moment bozunma katsayısı
    double epsilon;  // sayısal kararlılık sabiti

    struct State {
        Matrix m;   // 1. moment
        Matrix v;   // 2. moment
        int    t;   // adım sayacı

        State(int rows, int cols)
            : m(Matrix::zeros(rows, cols)),
              v(Matrix::zeros(rows, cols)),
              t(0) {}
    };

    std::unordered_map<std::string, State> states;

public:
    explicit Adam(double alpha   = 0.001,
                  double beta1   = 0.9,
                  double beta2   = 0.999,
                  double epsilon = 1e-8)
        : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    void step(Matrix& param, const Matrix& grad,
              const std::string& id) override {

        // İlk kez görülen parametre → state oluştur
        auto it = states.find(id);
        if (it == states.end()) {
            states.emplace(id, State(grad.rows, grad.cols));
            it = states.find(id);
        }

        State& s = it->second;
        s.t++;

        // m_t = β1*m + (1-β1)*g
        s.m = s.m * beta1 + grad * (1.0 - beta1);

        // v_t = β2*v + (1-β2)*g²
        s.v = s.v * beta2 + grad.hadamard(grad) * (1.0 - beta2);

        // Bias correction
        double bc1 = 1.0 - std::pow(beta1, s.t);
        double bc2 = 1.0 - std::pow(beta2, s.t);

        Matrix m_hat = s.m * (1.0 / bc1);
        Matrix v_hat = s.v * (1.0 / bc2);

        // θ = θ - α * m̂ / (√v̂ + ε)
        Matrix update = m_hat.hadamard(
            v_hat.apply([this](double vi){
                return 1.0 / (std::sqrt(vi) + epsilon);
            })
        ) * alpha;

        param -= update;
    }

    std::string name() const override {
        return "Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)";
    }
};
