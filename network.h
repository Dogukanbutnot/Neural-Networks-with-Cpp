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

public:
    // Optimizer geçmezsen varsayılan olarak Adam(α=0.001) kullanılır
    explicit NeuralNetwork(std::shared_ptr<Optimizer> opt = nullptr);

    // Katman ekle (ownership geçer)
    void add(std::unique_ptr<Layer> layer);

    // İleri geçiş
    Matrix predict(const Matrix& x);

    // Tek eğitim adımı — loss döner
    // loss_type: "mse" | "bce"
    double train_step(const Matrix& x, const Matrix& y,
                      const std::string& loss_type = "mse");

    // Tam eğitim döngüsü
    void fit(const std::vector<Matrix>& X_batches,
             const std::vector<Matrix>& Y_batches,
             int epochs,
             const std::string& loss_type = "mse",
             int print_every = 100);

    void print_summary() const;
};
