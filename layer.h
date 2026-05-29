#pragma once
#include "matrix.h"
#include "optimizer.h"
#include <memory>
#include <string>
#include <cmath>

// ============================================================
//  Abstract base
// ============================================================
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;

    // Optimizer pointer'ı ve katman ID'si ile güncelleme
    // Parametresi olmayan katmanlar (ReLU vb.) bu metodu override etmez.
    virtual void update(Optimizer* /*opt*/, const std::string& /*id*/) {}

    virtual std::string name() const = 0;
};

// ============================================================
//  Dense (Fully Connected) Layer
//  y = W*x + b
//  W: (out x in), b: (out x 1), x: (in x batch)
// ============================================================
class DenseLayer : public Layer {
    int in_size, out_size;
    Matrix W, b;        // ağırlıklar ve bias
    Matrix dW, db;      // gradyanlar (backward'dan gelir)
    Matrix input_cache; // backprop için girdi önbelleği

public:
    DenseLayer(int in_size, int out_size)
        : in_size(in_size), out_size(out_size),
          W(Matrix::random(out_size, in_size, 0.0, std::sqrt(2.0 / in_size))),
          b(Matrix::zeros(out_size, 1)),
          dW(Matrix::zeros(out_size, in_size)),
          db(Matrix::zeros(out_size, 1)),
          input_cache(1, 1)
    {}

    // x: (in x batch_size)  →  çıktı: (out x batch_size)
    Matrix forward(const Matrix& x) override {
        input_cache = x;
        int batch = x.cols;
        Matrix out = W * x;
        for (int i = 0; i < out.rows; i++)
            for (int j = 0; j < batch; j++)
                out.at(i, j) += b.at(i, 0);
        return out;
    }

    // grad_out: (out x batch)  →  grad_input: (in x batch)
    Matrix backward(const Matrix& grad_out) override {
        int batch = grad_out.cols;
        dW = grad_out * input_cache.transpose() * (1.0 / batch);
        db = grad_out.sum_cols() * (1.0 / batch);
        return W.transpose() * grad_out;
    }

    // Optimizer hangi algoritmayla güncelleneceğine kendisi karar verir.
    // id: "layer0_W", "layer0_b" gibi — her param için benzersiz.
    void update(Optimizer* opt, const std::string& id) override {
        opt->step(W, dW, id + "_W");
        opt->step(b, db, id + "_b");
    }

    std::string name() const override {
        return "Dense(" + std::to_string(in_size)
             + "->" + std::to_string(out_size) + ")";
    }
};

// ============================================================
//  Activation layers
// ============================================================
class ReLULayer : public Layer {
    Matrix mask; // 1 where input > 0

public:
    ReLULayer() : mask(1, 1) {}

    Matrix forward(const Matrix& x) override {
        mask = x.apply([](double v){ return v > 0.0 ? 1.0 : 0.0; });
        return x.apply([](double v){ return std::max(0.0, v); });
    }

    Matrix backward(const Matrix& grad_out) override {
        return grad_out.hadamard(mask);
    }

    std::string name() const override { return "ReLU"; }
};

class SigmoidLayer : public Layer {
    Matrix output_cache;

public:
    SigmoidLayer() : output_cache(1, 1) {}

    Matrix forward(const Matrix& x) override {
        output_cache = x.apply([](double v){ return 1.0 / (1.0 + std::exp(-v)); });
        return output_cache;
    }

    Matrix backward(const Matrix& grad_out) override {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        Matrix deriv = output_cache.hadamard(
            output_cache.apply([](double v){ return 1.0 - v; })
        );
        return grad_out.hadamard(deriv);
    }

    std::string name() const override { return "Sigmoid"; }
};

class TanhLayer : public Layer {
    Matrix output_cache;

public:
    TanhLayer() : output_cache(1, 1) {}

    Matrix forward(const Matrix& x) override {
        output_cache = x.apply([](double v){ return std::tanh(v); });
        return output_cache;
    }

    Matrix backward(const Matrix& grad_out) override {
        // tanh'(x) = 1 - tanh^2(x)
        Matrix deriv = output_cache.apply([](double v){ return 1.0 - v * v; });
        return grad_out.hadamard(deriv);
    }

    std::string name() const override { return "Tanh"; }
};

// ============================================================
//  Loss functions (standalone, not a layer)
// ============================================================
struct MSELoss {
    // pred, target: (out x batch)
    static double compute(const Matrix& pred, const Matrix& target) {
        Matrix diff = pred - target;
        return diff.hadamard(diff).sum() / (2.0 * diff.cols);
    }

    static Matrix gradient(const Matrix& pred, const Matrix& target) {
        return (pred - target) * (1.0 / pred.cols);
    }
};

struct BCELoss {
    // Binary cross-entropy for sigmoid outputs
    static double compute(const Matrix& pred, const Matrix& target) {
        Matrix p = pred.clip(1e-7, 1.0 - 1e-7);
        double loss = 0.0;
        for (int i = 0; i < p.rows * p.cols; i++) {
            double pi = p.data[i], ti = target.data[i];
            loss -= ti * std::log(pi) + (1.0 - ti) * std::log(1.0 - pi);
        }
        return loss / pred.cols;
    }

    static Matrix gradient(const Matrix& pred, const Matrix& target) {
        Matrix p = pred.clip(1e-7, 1.0 - 1e-7);
        Matrix denom = p.hadamard(p.apply([](double v){ return 1.0 - v; }));
        return (p - target).hadamard(
            denom.apply([](double v){ return 1.0 / v; })
        ) * (1.0 / pred.cols);
    }
};
