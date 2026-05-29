#include "network.h"
#include <stdexcept>
#include <iomanip>

NeuralNetwork::NeuralNetwork(std::shared_ptr<Optimizer> opt)
    : optimizer(opt ? opt : std::make_shared<Adam>())
{}

void NeuralNetwork::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Matrix NeuralNetwork::predict(const Matrix& x) {
    Matrix out = x;
    for (auto& layer : layers)
        out = layer->forward(out);
    return out;
}

double NeuralNetwork::train_step(const Matrix& x, const Matrix& y,
                                  const std::string& loss_type) {
    // ---- İleri geçiş ----
    Matrix pred = predict(x);

    // ---- Loss ve başlangıç gradyanı ----
    double loss;
    Matrix grad(1, 1);
    if (loss_type == "bce") {
        loss = BCELoss::compute(pred, y);
        grad = BCELoss::gradient(pred, y);
    } else {
        loss = MSELoss::compute(pred, y);
        grad = MSELoss::gradient(pred, y);
    }

    // ---- Geri geçiş ----
    for (int i = (int)layers.size() - 1; i >= 0; i--)
        grad = layers[i]->backward(grad);

    // ---- Ağırlık güncelleme — her katmana ID ver ----
    for (size_t i = 0; i < layers.size(); i++)
        layers[i]->update(optimizer.get(), "layer" + std::to_string(i));

    return loss;
}

void NeuralNetwork::fit(const std::vector<Matrix>& X_batches,
                         const std::vector<Matrix>& Y_batches,
                         int epochs,
                         const std::string& loss_type,
                         int print_every) {
    for (int epoch = 1; epoch <= epochs; epoch++) {
        double epoch_loss = 0.0;
        for (size_t b = 0; b < X_batches.size(); b++)
            epoch_loss += train_step(X_batches[b], Y_batches[b], loss_type);
        epoch_loss /= X_batches.size();

        if (epoch % print_every == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Loss: " << std::fixed
                      << std::setprecision(6) << epoch_loss << "\n";
        }
    }
}

void NeuralNetwork::print_summary() const {
    std::cout << "\n=== Network Summary ===\n";
    for (size_t i = 0; i < layers.size(); i++)
        std::cout << "  Layer " << i << ": " << layers[i]->name() << "\n";
    std::cout << "  Optimizer: " << optimizer->name() << "\n";
    std::cout << "======================\n\n";
}
