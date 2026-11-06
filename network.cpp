#include "network.h"
#include "conv.h"
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "serialization.h"

// loss_type: "mse" | "bce" | "ce"
static double loss_value(const std::string& t, const Matrix& p, const Matrix& y) {
    if (t == "ce")  return CrossEntropyLoss::compute(p, y);
    if (t == "bce") return BCELoss::compute(p, y);
    return MSELoss::compute(p, y);
}
static Matrix loss_grad(const std::string& t, const Matrix& p, const Matrix& y) {
    if (t == "ce")  return CrossEntropyLoss::gradient(p, y);
    if (t == "bce") return BCELoss::gradient(p, y);
    return MSELoss::gradient(p, y);
}

NeuralNetwork::NeuralNetwork(std::shared_ptr<Optimizer> opt)
    : optimizer(opt ? opt : std::make_shared<Adam>())
{}

void NeuralNetwork::add(std::unique_ptr<Layer> layer) {
    // Ilk katmanin girisi veri; ona giden gradyan hicbir yere gitmez
    if (layers.empty()) layer->set_compute_input_grad(false);
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
    double loss = loss_value(loss_type, pred, y);
    Matrix grad = loss_grad(loss_type, pred, y);

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
        total_loss += loss_value(loss_type, pred, Y_batches[b]);
    }
    return total_loss / X_batches.size();
}

double NeuralNetwork::fit(DataLoader& train_loader,
                          int epochs,
                          const std::string& loss_type,
                          int print_every,
                          DataLoader* val_loader) {
    if (early_stop && !val_loader)
        throw std::invalid_argument("Early stopping icin val_loader gerekli");

    double train_loss = 0.0;
    if (should_stop()) return train_loss;   // zaten durdurulmus: egitime devam etme

    for (int epoch = 1; epoch <= epochs; epoch++) {
        // Scheduler varsa bu epoch'un lr'sini ayarla (epoch_counter 0-tabanli)
        if (scheduler)
            optimizer->set_learning_rate(scheduler->lr_for_epoch(epoch_counter));

        set_training(true);
        train_loader.on_epoch_start();   // her epoch'ta yeniden karistir

        double loss_sum = 0.0;
        int    seen     = 0;
        for (int b = 0; b < train_loader.num_batches(); b++) {
            auto batch = train_loader.get_batch(b);
            int n = batch.first.cols;
            // loss batch ortalamasi -> ornek sayisiyla agirliklandir
            loss_sum += train_step(batch.first, batch.second, loss_type) * n;
            seen     += n;
        }
        train_loss = loss_sum / seen;

        // Validation loss: gerekirse bir kez hesaplanir (scheduler, early stopping, yazdirma paylasir)
        bool   have_vl = false;
        double vl      = 0.0;
        auto val_loss = [&]() {
            if (!have_vl) { vl = evaluate(*val_loader, loss_type); have_vl = true; }
            return vl;
        };

        // Metrige bakan scheduler'a (ReduceLROnPlateau) epoch sonu metrigi ver:
        // val loader varsa validation loss, yoksa egitim loss'u
        if (scheduler && scheduler->needs_metric())
            scheduler->on_epoch_end(val_loader ? val_loss() : train_loss);
        else if (scheduler)
            scheduler->on_epoch_end(train_loss);

        // Early stopping: izlenen metrige gore (val loss ya da val dogrulugu);
        // yeni en iyi epoch'ta agirliklari bellekte sakla
        double val_acc = 0.0;
        if (early_stop && early_stop->monitor() == Monitor::ValAccuracy)
            val_acc = accuracy(*val_loader);
        if (early_stop) {
            const double metric = (early_stop->monitor() == Monitor::ValAccuracy) ? val_acc : val_loss();
            if (early_stop->on_epoch_end(metric)) best_snapshot = snapshot();
        }
        epoch_counter++;

        if (print_every > 0 && (epoch % print_every == 0 || epoch == 1)) {
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Train Loss: " << std::fixed
                      << std::setprecision(6) << train_loss;
            if (val_loader)
                std::cout << " | Val Loss: " << val_loss();
            if (early_stop && early_stop->monitor() == Monitor::ValAccuracy)
                std::cout << " | Val Acc: " << std::setprecision(4) << val_acc << std::setprecision(6);
            if (scheduler)
                std::cout << " | LR: " << std::setprecision(6) << optimizer->learning_rate();
            std::cout << "\n";
        }

        if (early_stop && early_stop->should_stop()) {
            const bool restored = early_stop->restore_best() && !best_snapshot.empty();
            if (restored) restore(best_snapshot);
            if (print_every > 0)
                std::cout << "Early stopping: epoch " << early_stop->stopped_epoch() + 1
                          << "'te durdu (en iyi epoch " << early_stop->best_epoch() + 1
                          << (early_stop->monitor() == Monitor::ValAccuracy ? ", val dogrulugu " : ", val loss ")
                          << std::fixed << std::setprecision(6) << early_stop->best_value()
                          << (restored ? ", en iyi agirliklar geri yuklendi" : "") << ")\n";
            break;
        }
    }
    set_training(false);
    return train_loss;
}

double NeuralNetwork::evaluate(DataLoader& loader, const std::string& loss_type) {
    set_training(false);
    double loss_sum = 0.0;
    int    seen     = 0;
    for (int b = 0; b < loader.num_batches(); b++) {
        auto batch = loader.get_batch(b);
        Matrix pred = predict(batch.first);
        int n = batch.first.cols;
        loss_sum += loss_value(loss_type, pred, batch.second) * n;
        seen += n;
    }
    return loss_sum / seen;
}

double NeuralNetwork::accuracy(DataLoader& loader) {
    set_training(false);
    long correct = 0, total = 0;
    for (int b = 0; b < loader.num_batches(); b++) {
        auto batch = loader.get_batch(b);
        Matrix p = predict(batch.first);
        const Matrix& t = batch.second;
        for (int j = 0; j < p.cols; j++) {
            int pred, truth;
            if (p.rows == 1) {
                pred  = p.at(0, j) > 0.5 ? 1 : 0;
                truth = t.at(0, j) > 0.5 ? 1 : 0;
            } else {
                pred = truth = 0;
                for (int i = 1; i < p.rows; i++) {
                    if (p.at(i, j) > p.at(pred, j))  pred  = i;
                    if (t.at(i, j) > t.at(truth, j)) truth = i;
                }
            }
            correct += (pred == truth);
            total++;
        }
    }
    return total ? (double)correct / (double)total : 0.0;
}

double NeuralNetwork::weight_norm() const {
    double sum = 0.0;
    for (const auto& layer : layers)
        if (const auto* d = dynamic_cast<const DenseLayer*>(layer.get()))
            for (double w : d->weights().data) sum += w * w;
        else if (const auto* cv = dynamic_cast<const Conv2DLayer*>(layer.get()))
            for (double w : cv->weights().data) sum += w * w;
    return std::sqrt(sum);
}

void NeuralNetwork::print_summary() const {
    std::cout << "\n=== Network Summary ===\n";
    for (size_t i = 0; i < layers.size(); i++)
        std::cout << "  Layer " << i << ": " << layers[i]->name() << "\n";
    std::cout << "  Optimizer : " << optimizer->name() << "\n";
    std::cout << "======================\n\n";
}

// ------------------------------------------------------------
//  Model kaydet / yukle
//  Dosya: "NNCPP1" | int32 versiyon | int32 katman sayisi |
//         her katman icin: katman adi + parametreleri
// ------------------------------------------------------------
static const char   MODEL_MAGIC[6]  = {'N','N','C','P','P','1'};
static const int32_t MODEL_VERSION = 1;

void NeuralNetwork::save(std::ostream& os) const {
    os.write(MODEL_MAGIC, sizeof(MODEL_MAGIC));
    write_i32(os, MODEL_VERSION);
    write_i32(os, (int32_t)layers.size());
    for (const auto& layer : layers) {
        write_string(os, layer->name());
        layer->save(os);
    }
}

void NeuralNetwork::load(std::istream& is) {
    char magic[6] = {};
    is.read(magic, sizeof(magic));
    if (!is || std::string(magic, 6) != std::string(MODEL_MAGIC, 6))
        throw std::runtime_error("Gecerli bir model verisi degil");

    int32_t version = read_i32(is);
    if (version != MODEL_VERSION)
        throw std::runtime_error("Desteklenmeyen model versiyonu: " + std::to_string(version));

    int32_t count = read_i32(is);
    if (count != (int32_t)layers.size())
        throw std::runtime_error("Katman sayisi uyusmuyor: dosyada " + std::to_string(count)
                                 + ", agda " + std::to_string(layers.size()));

    for (size_t i = 0; i < layers.size(); i++) {
        std::string saved_name = read_string(is);
        if (saved_name != layers[i]->name())
            throw std::runtime_error("Katman " + std::to_string(i) + " uyusmuyor: dosyada '"
                                     + saved_name + "', agda '" + layers[i]->name() + "'");
        layers[i]->load(is);
    }
    set_training(false);   // yuklenen model cikarim (eval) modunda baslar
}

void NeuralNetwork::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("Dosya acilamadi (yazma): " + path);
    save(os);
    os.flush();
    if (!os) throw std::runtime_error("Yazma hatasi: " + path);
}

void NeuralNetwork::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Dosya acilamadi (okuma): " + path);
    try {
        load(is);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string(e.what()) + " [" + path + "]");
    }
}

std::string NeuralNetwork::snapshot() const {
    std::ostringstream os(std::ios::binary);
    save(os);
    return os.str();
}

void NeuralNetwork::restore(const std::string& snap) {
    std::istringstream is(snap, std::ios::binary);
    load(is);
}

void NeuralNetwork::restore_best_weights() {
    if (best_snapshot.empty())
        throw std::logic_error("restore_best_weights: kayitli en iyi agirlik yok "
                               "(early stopping ayarli degil ya da hic iyilesme olmadi)");
    restore(best_snapshot);
}
