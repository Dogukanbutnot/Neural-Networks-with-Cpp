#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

// ============================================================
//  Learning Rate Scheduler'lar
//
//  Fikir: egitimin basinda buyuk adimlarla hizli ilerle, sonlara dogru
//  adimi kucult ki agirliklar minimumun etrafinda "zipalamasin".
//
//  Arayuz (epoch'lar 0'dan baslar):
//    lr_for_epoch(e)   -> e. epoch'ta KULLANILACAK lr (epoch basinda cagrilir)
//    on_epoch_end(m)   -> epoch sonunda metrik (loss) ile cagrilir;
//                         sadece metrige bakan scheduler'lar kullanir
// ============================================================
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual double lr_for_epoch(int epoch) = 0;
    virtual bool   needs_metric() const { return false; }
    virtual void   on_epoch_end(double /*metric*/) {}
    virtual std::string name() const = 0;
};

namespace sched_detail {
    inline std::string num(double v) {
        std::ostringstream s; s << v; return s.str();
    }
}

// lr sabit (karsilastirma icin baseline)
class ConstantLR : public LRScheduler {
    double base;
public:
    explicit ConstantLR(double base_lr) : base(base_lr) {}
    double lr_for_epoch(int) override { return base; }
    std::string name() const override { return "Constant(" + sched_detail::num(base) + ")"; }
};

// Her step_size epoch'ta lr'yi gamma ile carp:  lr = base * gamma^(epoch / step_size)
class StepLR : public LRScheduler {
    double base, gamma; int step_size;
public:
    StepLR(double base_lr, int step_size, double gamma = 0.1)
        : base(base_lr), gamma(gamma), step_size(step_size) {
        if (step_size <= 0) throw std::invalid_argument("StepLR: step_size > 0 olmali");
    }
    double lr_for_epoch(int epoch) override {
        return base * std::pow(gamma, epoch / step_size);   // tamsayi bolme
    }
    std::string name() const override {
        return "Step(" + sched_detail::num(base) + ", every " + std::to_string(step_size)
             + ", x" + sched_detail::num(gamma) + ")";
    }
};

// Her epoch'ta gamma ile carp:  lr = base * gamma^epoch
class ExponentialLR : public LRScheduler {
    double base, gamma;
public:
    ExponentialLR(double base_lr, double gamma) : base(base_lr), gamma(gamma) {}
    double lr_for_epoch(int epoch) override { return base * std::pow(gamma, epoch); }
    std::string name() const override {
        return "Exp(" + sched_detail::num(base) + ", x" + sched_detail::num(gamma) + "/epoch)";
    }
};

// Kosinus egrisiyle base_lr'den eta_min'e pürüzsüz iner:
//   lr = eta_min + (base - eta_min) * (1 + cos(pi * t / T_max)) / 2 ,  t = min(epoch, T_max)
class CosineAnnealingLR : public LRScheduler {
    double base, eta_min; int T_max;
public:
    CosineAnnealingLR(double base_lr, int T_max, double eta_min = 0.0)
        : base(base_lr), eta_min(eta_min), T_max(T_max) {
        if (T_max <= 0) throw std::invalid_argument("CosineAnnealingLR: T_max > 0 olmali");
    }
    double lr_for_epoch(int epoch) override {
        const double pi = 3.14159265358979323846;
        const int t = std::min(epoch, T_max);
        return eta_min + (base - eta_min) * (1.0 + std::cos(pi * t / T_max)) / 2.0;
    }
    std::string name() const override {
        return "Cosine(" + sched_detail::num(base) + " -> " + sched_detail::num(eta_min)
             + ", T=" + std::to_string(T_max) + ")";
    }
};

// Metrik (loss) iyilesmeyi birakinca lr'yi kucultur.
//   - metric, best*(1 - threshold)'tan kucukse "iyilesme" sayilir
//   - patience'tan fazla ust uste iyilesmeyen epoch olursa lr *= factor
//   - lr min_lr'nin altina inmez
class ReduceLROnPlateau : public LRScheduler {
    double cur, factor, min_lr, threshold, best;
    int patience, bad_epochs;
public:
    ReduceLROnPlateau(double base_lr, double factor = 0.5, int patience = 2,
                      double min_lr = 1e-6, double threshold = 1e-4)
        : cur(base_lr), factor(factor), min_lr(min_lr), threshold(threshold),
          best(1e300), patience(patience), bad_epochs(0) {
        if (factor <= 0.0 || factor >= 1.0)
            throw std::invalid_argument("ReduceLROnPlateau: factor (0,1) araliginda olmali");
    }
    double lr_for_epoch(int) override { return cur; }
    bool needs_metric() const override { return true; }
    void on_epoch_end(double metric) override {
        if (metric < best * (1.0 - threshold)) {
            best = metric;
            bad_epochs = 0;
        } else if (++bad_epochs > patience) {
            cur = std::max(min_lr, cur * factor);
            bad_epochs = 0;
        }
    }
    std::string name() const override {
        return "Plateau(" + sched_detail::num(cur) + ", x" + sched_detail::num(factor)
             + ", patience=" + std::to_string(patience) + ")";
    }
};

// Isimden scheduler uret (main.cpp ve testler ayni ayarlari kullansin diye).
//   none | step | exp | cosine | plateau
// Hiperparametreler onceden sabitlendi (test sonuclarina bakilarak ayarlanmadi):
//   step   : her epochs/3 epoch'ta x0.3
//   exp    : epoch basina x0.8
//   cosine : lr0 -> 0, T = epochs
//   plateau: val loss 2 epoch iyilesmezse x0.5, taban 1e-5
inline std::shared_ptr<LRScheduler> make_scheduler(const std::string& kind, double lr0, int epochs) {
    if (kind == "none" || kind == "constant") return std::make_shared<ConstantLR>(lr0);
    if (kind == "step")    return std::make_shared<StepLR>(lr0, std::max(1, epochs / 3), 0.3);
    if (kind == "exp")     return std::make_shared<ExponentialLR>(lr0, 0.8);
    if (kind == "cosine")  return std::make_shared<CosineAnnealingLR>(lr0, std::max(1, epochs), 0.0);
    if (kind == "plateau") return std::make_shared<ReduceLROnPlateau>(lr0, 0.5, 1, 1e-5);
    throw std::invalid_argument("Bilinmeyen scheduler: " + kind
                                + " (none | step | exp | cosine | plateau)");
}
