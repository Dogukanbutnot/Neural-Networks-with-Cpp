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
//  Dropout Layer
//
//  Eğitimde: her nöronu p olasılıkla sıfırla,
//            kalanları 1/(1-p) ile scale et  ← inverted dropout
//  Testte:   hiçbir şey yapma, girdini olduğu gibi geçir
//
//  Neden inverted dropout?
//  Scaling'i eğitime taşırsak test sırasında ağa dokunmamıza gerek kalmaz.
//  Klasik dropout test'te scale yapardı — inverted bunun tersini yapar.
// ============================================================
class DropoutLayer : public Layer {
    double p;          // sıfırlama olasılığı (örn: 0.5)
    bool   training;   // true → dropout aktif, false → pass-through
    Matrix mask;       // backprop için önbelleklenen maske
    std::mt19937 rng;

public:
    explicit DropoutLayer(double drop_prob = 0.5)
        : p(drop_prob), training(true), mask(1, 1),
          rng(std::random_device{}())
    {
        if (p <= 0.0 || p >= 1.0)
            throw std::invalid_argument("Dropout p (0, 1) aralığında olmalı");
    }

    void set_training(bool mode) { training = mode; }

    // x: (features x batch)
    Matrix forward(const Matrix& x) override {
        if (!training) return x;   // test modunda dokunma

        // Bernoulli maske: 1/(1-p) ile scale edilmiş
        std::bernoulli_distribution dist(1.0 - p);
        double scale = 1.0 / (1.0 - p);

        mask = Matrix(x.rows, x.cols);
        for (auto& v : mask.data)
            v = dist(rng) ? scale : 0.0;

        return x.hadamard(mask);
    }

    // Gradyan aynı maske üzerinden geçer
    Matrix backward(const Matrix& grad_out) override {
        if (!training) return grad_out;
        return grad_out.hadamard(mask);
    }

    std::string name() const override {
        return "Dropout(p=" + std::to_string(p) + ")";
    }
};

// ============================================================
//  Batch Normalization Layer
//
//  Eğitimde:
//    μ  = mean(x)           batch ortalaması
//    σ² = var(x)            batch varyansı
//    x̂  = (x - μ) / √(σ²+ε)   normalize
//    y  = γ * x̂ + β         scale + shift  (öğrenilebilir)
//
//  Testte:
//    running_mean ve running_var kullanılır (eğitimde EMA ile biriktirilir)
//
//  Backprop:
//    dγ = sum(grad * x̂)
//    dβ = sum(grad)
//    dx = (1/N) * γ/σ * (N*grad - sum(grad) - x̂*sum(grad*x̂))
//
//  Neden BatchNorm kullanılır?
//  - Her katmanın girdi dağılımını sabit tutar (internal covariate shift azalır)
//  - Daha yüksek learning rate kullanmayı mümkün kılar
//  - Hafif bir regularization etkisi vardır
// ============================================================
class BatchNormLayer : public Layer {
    int    features;      // normalize edilecek boyut (satır sayısı)
    double eps;           // sayısal kararlılık için küçük sabit
    double momentum;      // running stats için EMA katsayısı
    bool   training;

    // Öğrenilebilir parametreler
    Matrix gamma;         // scale  (features x 1), başlangıç: 1
    Matrix beta;          // shift  (features x 1), başlangıç: 0
    Matrix dgamma, dbeta; // gradyanlar

    // Test için biriktirilen istatistikler
    Matrix running_mean;
    Matrix running_var;

    // Backprop için önbellek
    Matrix x_hat;         // normalize edilmiş x
    Matrix x_centered;    // x - mean
    Matrix std_inv;       // 1 / sqrt(var + eps)  (features x 1)
    int    batch_size;

public:
    explicit BatchNormLayer(int features, double eps = 1e-5, double momentum = 0.1)
        : features(features), eps(eps), momentum(momentum), training(true),
          gamma(Matrix::ones(features, 1)),
          beta(Matrix::zeros(features, 1)),
          dgamma(Matrix::zeros(features, 1)),
          dbeta(Matrix::zeros(features, 1)),
          running_mean(Matrix::zeros(features, 1)),
          running_var(Matrix::ones(features, 1)),
          x_hat(1, 1), x_centered(1, 1),
          std_inv(Matrix::ones(features, 1)),
          batch_size(1)
    {}

    void set_training(bool mode) { training = mode; }

    // x: (features x batch)  →  y: (features x batch)
    Matrix forward(const Matrix& x) override {
        batch_size = x.cols;

        Matrix mean(features, 1);   // μ  (features x 1)
        Matrix var(features, 1);    // σ² (features x 1)

        if (training) {
            // Batch ortalaması: her feature için batch boyunca ortalama
            for (int i = 0; i < features; i++) {
                double s = 0.0;
                for (int j = 0; j < batch_size; j++) s += x.at(i, j);
                mean.at(i, 0) = s / batch_size;
            }

            // Batch varyansı
            for (int i = 0; i < features; i++) {
                double s = 0.0;
                for (int j = 0; j < batch_size; j++) {
                    double d = x.at(i, j) - mean.at(i, 0);
                    s += d * d;
                }
                var.at(i, 0) = s / batch_size;
            }

            // Running stats güncelle (EMA)
            for (int i = 0; i < features; i++) {
                running_mean.at(i, 0) = (1 - momentum) * running_mean.at(i, 0)
                                      + momentum * mean.at(i, 0);
                running_var.at(i, 0)  = (1 - momentum) * running_var.at(i, 0)
                                      + momentum * var.at(i, 0);
            }
        } else {
            // Test modunda running stats kullan
            mean = running_mean;
            var  = running_var;
        }

        // std_inv = 1 / sqrt(σ² + ε)
        std_inv = var.apply([this](double v){ return 1.0 / std::sqrt(v + eps); });

        // x_centered = x - μ  (broadcast)
        x_centered = Matrix(features, batch_size);
        for (int i = 0; i < features; i++)
            for (int j = 0; j < batch_size; j++)
                x_centered.at(i, j) = x.at(i, j) - mean.at(i, 0);

        // x̂ = x_centered * std_inv  (broadcast)
        x_hat = Matrix(features, batch_size);
        for (int i = 0; i < features; i++)
            for (int j = 0; j < batch_size; j++)
                x_hat.at(i, j) = x_centered.at(i, j) * std_inv.at(i, 0);

        // y = γ * x̂ + β  (broadcast)
        Matrix out(features, batch_size);
        for (int i = 0; i < features; i++)
            for (int j = 0; j < batch_size; j++)
                out.at(i, j) = gamma.at(i, 0) * x_hat.at(i, j) + beta.at(i, 0);

        return out;
    }

    // grad_out: (features x batch)  →  grad_input: (features x batch)
    Matrix backward(const Matrix& grad_out) override {
        int N = batch_size;

        // dγ = sum_batch(grad * x̂)
        // dβ = sum_batch(grad)
        for (int i = 0; i < features; i++) {
            double dg = 0.0, db = 0.0;
            for (int j = 0; j < N; j++) {
                dg += grad_out.at(i, j) * x_hat.at(i, j);
                db += grad_out.at(i, j);
            }
            dgamma.at(i, 0) = dg;
            dbeta.at(i, 0)  = db;
        }

        // dx̂ = grad * γ
        Matrix dx_hat(features, N);
        for (int i = 0; i < features; i++)
            for (int j = 0; j < N; j++)
                dx_hat.at(i, j) = grad_out.at(i, j) * gamma.at(i, 0);

        // Tam backprop formülü (Ioffe & Szegedy, 2015):
        // dx = (1/N) * std_inv * (N*dx̂ - sum(dx̂) - x̂*sum(dx̂*x̂))
        Matrix dx(features, N);
        for (int i = 0; i < features; i++) {
            double sum_dxh  = 0.0;
            double sum_dxhx = 0.0;
            for (int j = 0; j < N; j++) {
                sum_dxh  += dx_hat.at(i, j);
                sum_dxhx += dx_hat.at(i, j) * x_hat.at(i, j);
            }
            for (int j = 0; j < N; j++) {
                dx.at(i, j) = std_inv.at(i, 0) / N *
                    (N * dx_hat.at(i, j) - sum_dxh
                     - x_hat.at(i, j) * sum_dxhx);
            }
        }

        return dx;
    }

    // γ ve β'yı optimizer ile güncelle
    void update(Optimizer* opt, const std::string& id) override {
        opt->step(gamma, dgamma, id + "_gamma");
        opt->step(beta,  dbeta,  id + "_beta");
    }

    std::string name() const override {
        return "BatchNorm(" + std::to_string(features) + ")";
    }
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
