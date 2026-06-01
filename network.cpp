#include "network.h"
#include <stdexcept>
#include <iomanip>

NeuralNetwork::NeuralNetwork(std::shared_ptr<Optimizer> opt)
    : optimizer(opt ? opt : std::make_shared<Adam>())
{}

void NeuralNetwork::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

// Tüm katmanlara training/eval modunu bildir
// dynamic_cast ile sadece ilgili katmanlar etkilenir
void NeuralNetwork::set_training(bool mode) {
    training_mode = mode;
    for (auto& layer : layers) {
        if (auto* d = dynamic_cast<DropoutLayer*>(layer.get()))
            d->set_training(mode);
        if (auto* b = dynamic_cast<BatchNormLayer*>(layer.get()))
            b->set_training(mode);
    }
}

Matrix NeuralNetwork::predict(const Matrix& x) {
    Matrix out = x;
    for (auto& layer : layers)
        out = layer->forward(out);
    return out;
}

double NeuralNetwork::train_step(const Matrix& x, const Matrix& y,
                                  const std::string& loss_type) {
    // Eğitim modunu garantile
    if (!training_mode) set_training(true);

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

    // ---- Ağırlık güncelleme ----
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
        // Eğitim adımı
        set_training(true);
        double train_loss = 0.0;
        for (size_t b = 0; b < X_batches.size(); b++)
            train_loss += train_step(X_batches[b], Y_batches[b], loss_type);
        train_loss /= X_batches.size();

        if (epoch % print_every == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Train Loss: " << std::fixed
                      << std::setprecision(6) << train_loss << "\n";
        }
    }
    // Eğitim bitti — eval moduna geç
    set_training(false);
}

double NeuralNetwork::evaluate(const std::vector<Matrix>& X_batches,
                                const std::vector<Matrix>& Y_batches,
                                const std::string& loss_type) {
    set_training(false);
    double total_loss = 0.0;
    for (size_t b = 0; b < X_batches.size(); b++) {
        Matrix pred = predict(X_batches[b]);
        if (loss_type == "bce")
            total_loss += BCELoss::compute(pred, Y_batches[b]);
        else
            total_loss += MSELoss::compute(pred, Y_batches[b]);
    }
    return total_loss / X_batches.size();
}

void NeuralNetwork::print_summary() const {
    std::cout << "\n=== Network Summary ===\n";
    for (size_t i = 0; i < layers.size(); i++)
        std::cout << "  Layer " << i << ": " << layers[i]->name() << "\n";
    std::cout << "  Optimizer : " << optimizer->name() << "\n";
    std::cout << "======================\n\n";
}
