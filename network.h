#pragma once
#include "layer.h"
#include "optimizer.h"
#include "dataloader.h"
#include "scheduler.h"
#include "earlystopping.h"
#include <iosfwd>
#include <vector>
#include <memory>
#include <string>
#include <iostream>

class NeuralNetwork {
    std::vector<std::unique_ptr<Layer>> layers;
    std::shared_ptr<Optimizer> optimizer;
    bool training_mode = true;
    std::shared_ptr<LRScheduler> scheduler;   // istege bagli
    int  epoch_counter = 0;                   // fit() cagrilari boyunca korunur
    std::shared_ptr<EarlyStopping> early_stop;   // istege bagli
    std::string best_snapshot;                   // en iyi epoch'un agirliklari (bellekte)

public:
    explicit NeuralNetwork(std::shared_ptr<Optimizer> opt = nullptr);

    void add(std::unique_ptr<Layer> layer);

    // Tüm Dropout ve BatchNorm katmanlarını eğitim/test moduna al
    void set_training(bool mode);

    // İleri geçiş (mevcut mod ne ise onu kullanır)
    Matrix predict(const Matrix& x);

    // loss_type: "mse" | "bce" | "ce"
    double train_step(const Matrix& x, const Matrix& y,
                      const std::string& loss_type = "mse");

    void fit(const std::vector<Matrix>& X_batches,
             const std::vector<Matrix>& Y_batches,
             int epochs,
             const std::string& loss_type = "mse",
             int print_every = 100);

    // Validation loss hesapla (test modunda — dropout kapalı)
    double evaluate(const std::vector<Matrix>& X_batches,
                    const std::vector<Matrix>& Y_batches,
                    const std::string& loss_type = "mse");

    // ---- DataLoader tabanli API (mini-batch) ----
    // val_loader verilirse her print_every epoch'ta validation loss da yazdirilir
    // Son epoch'un egitim loss'unu dondurur. print_every <= 0 ise hic yazdirmaz.
    double fit(DataLoader& train_loader,
               int epochs,
               const std::string& loss_type = "mse",
               int print_every = 10,
               DataLoader* val_loader = nullptr);

    double evaluate(DataLoader& loader,
                    const std::string& loss_type = "mse");

    // Dogruluk (0..1 orani), test modunda.
    //  - cikis satiri > 1 : cok sinifli, tahmin ve hedefin argmax'i karsilastirilir (hedef one-hot)
    //  - cikis satiri = 1 : ikili, (tahmin > 0.5) ile (hedef > 0.5) karsilastirilir
    double accuracy(DataLoader& loader);

    // Agirliklari ikili dosyaya yaz / dosyadan yukle.
    // load() icin ayni mimariyle kurulmus bir ag gerekir; katman sayisi ve
    // katman adlari (boyutlar dahil) uyusmazsa std::runtime_error firlatir.
    void save(const std::string& path) const;
    void load(const std::string& path);
    void save(std::ostream& os) const;      // ayni format, akisa
    void load(std::istream& is);

    // Bellekte anlik goruntu: tum agirliklar + BatchNorm istatistikleri
    std::string snapshot() const;
    void restore(const std::string& snap);

    // Scheduler ver: DataLoader'li fit() her epoch basinda lr'yi ondan alir.
    // (nullptr = sabit lr). Epoch sayaci sifirlanir.
    void set_scheduler(std::shared_ptr<LRScheduler> s) { scheduler = std::move(s); epoch_counter = 0; }
    double learning_rate() const { return optimizer->learning_rate(); }

    // Early stopping: DataLoader'li fit() her epoch sonunda validation loss'u ona
    // verir (val_loader ZORUNLU). Yeni en iyi epoch'ta agirliklar bellekte saklanir.
    // Durma tetiklenirse (ve restore_best aciksa) en iyi agirliklar geri yuklenir.
    void set_early_stopping(std::shared_ptr<EarlyStopping> es) { early_stop = std::move(es); best_snapshot.clear(); }
    bool should_stop() const { return early_stop && early_stop->should_stop(); }
    void restore_best_weights();    // en iyi epoch'un agirliklarini geri yukle

    // Tum Dense ve Conv2D agirliklarinin (bias haric) birlesik L2 normu: sqrt(sum W^2)
    double weight_norm() const;

    void print_summary() const;
};
