#pragma once
#include "layer.h"
#include "optimizer.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>

class NeuralNetwork {
    std::vector<std::unique_ptr<Layer>> layers;
    std::shared_ptr<Optimizer> optimizer;
    bool training_mode = true;

public:
    explicit NeuralNetwork(std::shared_ptr<Optimizer> opt = nullptr);

    void add(std::unique_ptr<Layer> layer);

    // Tüm Dropout ve BatchNorm katmanlarını eğitim/test moduna al
    void set_training(bool mode);

    // İleri geçiş (mevcut mod ne ise onu kullanır)
    Matrix predict(const Matrix& x);

    // loss_type: "mse" | "bce"
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

    void print_summary() const;
};
