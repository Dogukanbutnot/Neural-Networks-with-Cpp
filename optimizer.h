#pragma once
#include "matrix.h"
#include <unordered_map>
#include <string>
#include <cmath>
#include <sstream>

// ============================================================
//  Weight decay hangi parametrelere uygulanir?
//  Sadece agirlik matrislerine (id "..._W" ile biter). Bias ve BatchNorm
//  gamma/beta'ya uygulanmaz: bunlari sifira cekmek modeli sinirlar ama
//  ezberlemeyi azaltmaz (standart uygulama).
// ============================================================
inline bool decays_param(const std::string& id) {
    return id.size() >= 2 && id.compare(id.size() - 2, 2, "_W") == 0;
}

// ============================================================
//  Optimizer abstract base
// ============================================================
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Matrix& param, const Matrix& grad,
                      const std::string& param_id) = 0;
    // Scheduler'larin kullanmasi icin ogrenme orani erisimi
    virtual void   set_learning_rate(double lr) = 0;
    virtual double learning_rate() const = 0;
    virtual std::string name() const = 0;
};

// ============================================================
//  SGD  (karşılaştırma için)
// ============================================================
class SGD : public Optimizer {
    double lr;
    double weight_decay;   // L2: gradyana wd*theta eklenir (0 = kapali)
public:
    explicit SGD(double lr = 0.01, double weight_decay = 0.0)
        : lr(lr), weight_decay(weight_decay) {}

    void step(Matrix& param, const Matrix& grad,
              const std::string& id) override {
        if (weight_decay != 0.0 && decays_param(id)) {
            // theta -= lr * (g + wd*theta)
            for (size_t i = 0; i < param.data.size(); i++)
                param.data[i] -= (grad.data[i] + weight_decay * param.data[i]) * lr;
        } else {
            param -= grad * lr;
        }
    }

    void   set_learning_rate(double v) override { lr = v; }
    double learning_rate() const override { return lr; }

    std::string name() const override {
        std::ostringstream ss;
        ss << "SGD(lr=" << lr;
        if (weight_decay != 0.0) ss << ", L2=" << weight_decay;
        ss << ")";
        return ss.str();
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
//
//  Weight decay (yalnizca "..._W" agirliklarina; bkz. decays_param):
//    decoupled = false  ->  L2: g <- g + wd*θ, moment'lere GIRER.
//                           Adam gradyani √v̂ ile boldugu icin buyuk gradyanli
//                           agirliklarda decay zayiflar (etkisi ongorulemez).
//    decoupled = true   ->  AdamW (Loshchilov & Hutter): θ <- θ*(1 - α*wd)
//                           once uygulanir, sonra normal Adam adimi. Decay,
//                           gradyan buyuklugunden bagimsiz ve lr ile olceklenir
//                           (scheduler lr'yi dusurdukce decay de azalir).
// ============================================================
class Adam : public Optimizer {
    double alpha;    // öğrenme oranı
    double beta1;    // 1. moment bozunma katsayısı
    double beta2;    // 2. moment bozunma katsayısı
    double epsilon;  // sayısal kararlılık sabiti
    double weight_decay;   // 0 = kapali
    bool   decoupled;      // true: AdamW, false: L2

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
                  double epsilon = 1e-8,
                  double weight_decay = 0.0,
                  bool   decoupled    = false)
        : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon),
          weight_decay(weight_decay), decoupled(decoupled) {}

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

        // Bias correction faktorleri
        const double bc1 = 1.0 - std::pow(beta1, s.t);
        const double bc2 = 1.0 - std::pow(beta2, s.t);
        const double inv_bc1 = 1.0 / bc1;
        const double inv_bc2 = 1.0 / bc2;
        const double one_m_b1 = 1.0 - beta1;
        const double one_m_b2 = 1.0 - beta2;

        // Tek dongude birlesik guncelleme (gecici matris yok).
        // Islem sirasi onceki Matrix-islemli surumle birebir ayni tutuldu,
        // bu yuzden sonuclar bit-bit aynidir.
        //   m  = m*b1 + g*(1-b1)
        //   v  = v*b2 + (g*g)*(1-b2)
        //   th = th - ((m/bc1) * (1/(sqrt(v/bc2)+eps))) * alpha
        const bool   decay  = weight_decay != 0.0 && decays_param(id);
        const bool   use_l2 = decay && !decoupled;
        const double shrink = (decay && decoupled) ? (1.0 - alpha * weight_decay) : 1.0;

        const size_t n = grad.data.size();
        double*       __restrict p  = param.data.data();
        const double* __restrict g  = grad.data.data();
        double*       __restrict m  = s.m.data.data();
        double*       __restrict v  = s.v.data.data();
        for (size_t i = 0; i < n; i++) {
            if (shrink != 1.0) p[i] *= shrink;                       // AdamW
            const double gi = use_l2 ? g[i] + weight_decay * p[i]    // L2
                                     : g[i];
            m[i] = m[i] * beta1 + gi * one_m_b1;
            v[i] = v[i] * beta2 + (gi * gi) * one_m_b2;
            const double m_hat = m[i] * inv_bc1;
            const double v_hat = v[i] * inv_bc2;
            p[i] -= (m_hat * (1.0 / (std::sqrt(v_hat) + epsilon))) * alpha;
        }
    }

    void   set_learning_rate(double v) override { alpha = v; }
    double learning_rate() const override { return alpha; }

    std::string name() const override {
        std::ostringstream ss;
        ss << (decoupled && weight_decay != 0.0 ? "AdamW" : "Adam")
           << "(alpha=" << alpha << ", beta1=" << beta1
           << ", beta2=" << beta2 << ", eps=" << epsilon;
        if (weight_decay != 0.0)
            ss << (decoupled ? ", decoupled wd=" : ", L2=") << weight_decay;
        ss << ")";
        return ss.str();
    }
};

// AdamW kisayolu: ayrik (decoupled) weight decay'li Adam
class AdamW : public Adam {
public:
    explicit AdamW(double alpha        = 0.001,
                   double weight_decay = 0.01,
                   double beta1        = 0.9,
                   double beta2        = 0.999,
                   double epsilon      = 1e-8)
        : Adam(alpha, beta1, beta2, epsilon, weight_decay, true) {}
};
