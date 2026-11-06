#include "network.h"
#include "dataloader.h"
#include "mnist.h"
#include "conv.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

// Dogruluk (%): buyuk matrisleri tek seferde vermemek icin 1000'lik parcalarla
double accuracy(NeuralNetwork& net, const Matrix& X, const std::vector<int>& labels) {
    net.set_training(false);
    const int CHUNK = 1000;
    int ok = 0;
    for (int s = 0; s < X.cols; s += CHUNK) {
        int e = std::min(s + CHUNK, X.cols);
        Matrix p = net.predict(slice_columns(X, s, e));
        for (int j = 0; j < p.cols; j++) {
            int best = 0;
            for (int i = 1; i < p.rows; i++)
                if (p.at(i, j) > p.at(best, j)) best = i;
            if (best == labels[s + j]) ok++;
        }
    }
    return 100.0 * ok / X.cols;
}

// Varyantlar:
//   base      : Dense-ReLU (onceki model)
//   drop      : Dense-ReLU-Dropout(0.2)
//   bn        : Dense-BatchNorm-ReLU
//   bn_drop   : Dense-BatchNorm-ReLU-Dropout(0.2)
// Optimizer secimi:  adam | l2=<lambda> | adamw=<wd>      (ornek: adamw=0.05)
//   adam      : duz Adam
//   l2=1e-4   : Adam + L2 (gradyana lambda*W eklenir)
//   adamw=0.05: AdamW (ayrik weight decay)
std::shared_ptr<Optimizer> make_optimizer(const std::string& spec, double lr0) {
    if (spec == "adam") return std::make_shared<Adam>(lr0, 0.9, 0.999, 1e-8);
    const size_t eq = spec.find('=');
    if (eq != std::string::npos) {
        const std::string kind = spec.substr(0, eq);
        const double v = std::stod(spec.substr(eq + 1));
        if (kind == "l2")    return std::make_shared<Adam>(lr0, 0.9, 0.999, 1e-8, v, false);
        if (kind == "adamw") return std::make_shared<AdamW>(lr0, v, 0.9, 0.999, 1e-8);
    }
    throw std::invalid_argument("Bilinmeyen optimizer: " + spec + " (adam | l2=<lambda> | adamw=<wd>)");
}

NeuralNetwork make_net(const std::string& variant, std::shared_ptr<Optimizer> opt, unsigned seed) {
    // ---- CNN varyantlari ----
    //   cnn      : Conv(1->8)-ReLU-Pool - Conv(8->16)-ReLU-Pool - Dense(784->64)-ReLU - Dense(64->10)
    //   cnn_drop : ayni, Dense'ten sonra Dropout(0.2)
    if (variant == "cnn" || variant == "cnn_drop") {
        NeuralNetwork net(opt);
        net.add(std::make_unique<Conv2DLayer>(1, 28, 28, 8, 3, 1, 1));         // 8x28x28
        net.add(std::make_unique<ReLULayer>());
        net.add(std::make_unique<MaxPool2DLayer>(8, 28, 28, 2));              // 8x14x14
        net.add(std::make_unique<Conv2DLayer>(8, 14, 14, 16, 3, 1, 1));        // 16x14x14
        net.add(std::make_unique<ReLULayer>());
        net.add(std::make_unique<MaxPool2DLayer>(16, 14, 14, 2));             // 16x7x7 = 784
        net.add(std::make_unique<DenseLayer>(16 * 7 * 7, 64));
        net.add(std::make_unique<ReLULayer>());
        if (variant == "cnn_drop")
            net.add(std::make_unique<DropoutLayer>(0.2, (seed == 42) ? 1u : seed * 2654435761u));
        net.add(std::make_unique<DenseLayer>(64, 10));
        net.add(std::make_unique<SoftmaxLayer>());
        return net;
    }

    bool use_bn   = (variant == "bn"   || variant == "bn_drop");
    bool use_drop = (variant == "drop" || variant == "bn_drop");
    if (!use_bn && !use_drop && variant != "base")
        throw std::invalid_argument("Bilinmeyen varyant: " + variant
                                    + " (base | drop | bn | bn_drop | cnn | cnn_drop)");

    NeuralNetwork net(opt);
    const int sizes[] = {784, 128, 64};
    // Dropout tohumlari: seed=42 (varsayilan) eski deneylerdeki 1,2 degerlerini
    // korur, boylece onceki sonuclar birebir tekrarlanabilir.
    unsigned dseed = (seed == 42) ? 1u : seed * 2654435761u;
    for (int i = 0; i < 2; i++) {
        net.add(std::make_unique<DenseLayer>(sizes[i], sizes[i + 1]));
        if (use_bn)   net.add(std::make_unique<BatchNormLayer>(sizes[i + 1]));
        net.add(std::make_unique<ReLULayer>());
        if (use_drop) net.add(std::make_unique<DropoutLayer>(0.2, dseed++));
    }
    net.add(std::make_unique<DenseLayer>(64, 10));
    net.add(std::make_unique<SoftmaxLayer>());
    return net;
}

// Kullanim:  ./nn [veri_klasoru] [epoch] [batch] [max_egitim_ornegi] [varyant]
//                 [scheduler] [lr0] [tohum] [optimizer] [patience] [izleme]
//   scheduler: none | step | exp | cosine | plateau   (varsayilan none)
//   lr0      : baslangic ogrenme orani                 (varsayilan 0.001)
//   tohum    : agirlik baslatma + karistirma + dropout (varsayilan 42)
//   optimizer: adam | l2=<lambda> | adamw=<wd>         (varsayilan adam)
//   patience : early stopping sabri (izlenen metrik kac epoch iyilesmezse dur); 0 = kapali
//   izleme   : loss | acc  -> early stopping val loss'u mu val dogrulugunu mu izlesin (varsayilan loss)
//              (patience=0 iken de "en iyi epoch" bu metrige gore secilir)
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    int epochs      = argc > 2 ? std::stoi(argv[2]) : 10;
    int batch       = argc > 3 ? std::stoi(argv[3]) : 64;
    int max_train   = argc > 4 ? std::stoi(argv[4]) : 0;   // 0 = hepsi
    std::string variant = argc > 5 ? argv[5] : "base";
    std::string sched_name = argc > 6 ? argv[6] : "none";
    double lr0      = argc > 7 ? std::stod(argv[7]) : 0.001;
    unsigned seed   = argc > 8 ? (unsigned)std::stoul(argv[8]) : 42u;
    std::string opt_spec = argc > 9 ? argv[9] : "adam";
    int patience = argc > 10 ? std::stoi(argv[10]) : 0;
    std::string monitor_name = argc > 11 ? argv[11] : "loss";
    if (monitor_name != "loss" && monitor_name != "acc")
        throw std::invalid_argument("Bilinmeyen izleme modu: " + monitor_name + " (loss | acc)");
    const bool monitor_acc = (monitor_name == "acc");
    const std::string model_path = "mnist_" + variant + "_" + sched_name + "_" + opt_spec + "_s" + std::to_string(seed)
                                 + (patience > 0 ? "_p" + std::to_string(patience) : std::string())
                                 + (monitor_acc ? "_acc" : std::string()) + ".nn";
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';

    try {
        // ---------- Veri ----------
        std::cout << "MNIST yukleniyor: " << dir << "\n";
        Matrix Xall = load_mnist_images(dir + "train-images-idx3-ubyte", max_train);
        std::vector<int> lall = load_mnist_labels(dir + "train-labels-idx1-ubyte", max_train);
        Matrix Xte = load_mnist_images(dir + "t10k-images-idx3-ubyte");
        std::vector<int> lte = load_mnist_labels(dir + "t10k-labels-idx1-ubyte");

        // egitim / dogrulama ayrimi: son 5000 ornek dogrulama
        int n_val = std::min(5000, Xall.cols / 10);
        int n_tr  = Xall.cols - n_val;
        Matrix Xtr = slice_columns(Xall, 0, n_tr);
        Matrix Xva = slice_columns(Xall, n_tr, Xall.cols);
        std::vector<int> ltr = slice_labels(lall, 0, n_tr);
        std::vector<int> lva = slice_labels(lall, n_tr, Xall.cols);
        std::cout << "  egitim: " << Xtr.cols << "  dogrulama: " << Xva.cols
                  << "  test: " << Xte.cols << "  (piksel: " << Xtr.rows << ")\n\n";

        Matrix Ytr = one_hot(ltr, 10), Yva = one_hot(lva, 10);

        // ---------- Ag ----------
        Matrix::set_seed(seed);                       // agirlik baslatma tohumu
        NeuralNetwork net = make_net(variant, make_optimizer(opt_spec, lr0), seed);
        net.set_scheduler(make_scheduler(sched_name, lr0, epochs));
        net.print_summary();
        std::cout << "Scheduler: " << make_scheduler(sched_name, lr0, epochs)->name()
                  << "  | tohum: " << seed << "\n";
        DataLoader train(Xtr, Ytr, batch, true, false, seed);
        DataLoader val(Xva, Yva, 1000, false);

        std::cout << "Batch: " << batch << "  adim/epoch: " << train.num_batches() << "\n";
        std::cout << "Baslangic dogrulama dogrulugu (egitimsiz): "
                  << std::fixed << std::setprecision(2) << accuracy(net, Xva, lva) << "%\n\n";

        // ---------- Egitim: her epoch sonunda dogrulama + early stopping ----------
        // Early stopping kapaliyken (patience=0) yine de en iyi epoch izlenir; sadece hic durdurmaz.
        // restore_best=false: once SON epoch'u olcmek, sonra elle en iyiye donmek icin.
        auto es = std::make_shared<EarlyStopping>(patience > 0 ? patience : epochs + 1, 0.0, /*restore_best=*/false,
                                                  monitor_acc ? Monitor::ValAccuracy : Monitor::ValLoss);
        net.set_early_stopping(es);

        double last_val_loss = 0.0;
        int    epochs_run    = 0;
        std::vector<double> vls, vas;       // her epoch'un val loss'u ve val dogrulugu (%)
        auto t_start = std::chrono::steady_clock::now();
        for (int e = 1; e <= epochs && !net.should_stop(); e++) {
            auto t0 = std::chrono::steady_clock::now();
            double tr_loss = net.fit(train, 1, "ce", 0, &val);    // sessiz, 1 epoch
            double va_loss = net.evaluate(val, "ce");
            double va_acc  = accuracy(net, Xva, lva);
            vls.push_back(va_loss); vas.push_back(va_acc);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            epochs_run    = e;
            last_val_loss = va_loss;
            bool improved = (es->best_epoch() == e - 1);
            std::cout << "Epoch " << std::setw(2) << e << "/" << epochs
                      << " | Train Loss: " << std::fixed << std::setprecision(4) << tr_loss
                      << " | Val Loss: " << va_loss
                      << " | Val Acc: " << std::setprecision(2) << va_acc << "%"
                      << " | LR: " << std::setprecision(5) << net.learning_rate()
                      << " | " << std::setprecision(1) << sec << " sn"
                      << (improved ? "  * en iyi" : "") << "\n";
        }
        double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        const bool stopped_early = es->should_stop() && patience > 0;
        const int  best_epoch    = es->best_epoch() + 1;               // 1-tabanli
        // Secilen (en iyi) epoch'taki val loss ve val dogrulugu (izleme modundan bagimsiz raporlanir)
        const double best_val_loss = vls[es->best_epoch()];
        const double best_val_acc  = vas[es->best_epoch()];
        if (stopped_early)
            std::cout << "\nEarly stopping: epoch " << epochs_run << "'te durdu (sabir " << patience
                      << "), en iyi epoch " << best_epoch << "\n";

        // Son epoch modelinin test dogrulugu (en iyiye donmeden ONCE olcuyoruz)
        double test_last = accuracy(net, Xte, lte);
        double wnorm_last = net.weight_norm();
        double train_last = accuracy(net, Xtr, ltr);

        // En iyi epoch'a don (son epoch ezberlemis olabilir) ve dosyaya kaydet
        net.restore_best_weights();
        net.save(model_path);
        std::cout << "\nEn iyi model: epoch " << best_epoch
                  << " (val loss " << std::setprecision(4) << best_val_loss
                  << ", val dogrulugu " << std::setprecision(2) << best_val_acc << "%; secim olcutu: "
                  << (monitor_acc ? "dogruluk" : "loss") << ") yuklendi\n";

        // ---------- Final: test seti SADECE burada, bir kez ----------
        double train_best = accuracy(net, Xtr, ltr);
        double val_best   = accuracy(net, Xva, lva);
        double test_best  = accuracy(net, Xte, lte);
        std::cout << "\n=== Sonuc [" << variant << "] ===\n" << std::fixed << std::setprecision(2)
                  << "  SON epoch (" << epochs_run << ")  : egitim " << train_last
                  << "%  | test " << test_last << "%\n"
                  << "  EN IYI epoch (" << best_epoch << "): egitim " << train_best
                  << "%  | dogrulama " << val_best << "%  | test " << test_best << "%\n"
                  << "  Egitim-test acigi (son epoch): " << (train_last - test_last) << " puan\n"
                  << "  Toplam sure: " << std::setprecision(1) << total << " sn\n"
                  << "  Model dosyasi: " << model_path << "\n";
        // Betikle toplamak icin tek satirlik ozet
        std::cout << std::setprecision(4)
                  << "RESULT variant=" << variant << " sched=" << sched_name << " lr0=" << lr0
                  << " seed=" << seed << " opt=" << opt_spec << " wnorm=" << std::setprecision(2) << wnorm_last << std::setprecision(4)
                  << " last_test=" << std::setprecision(2) << test_last
                  << " best_test=" << test_best << " gap=" << (train_last - test_last)
                  << " last_val_loss=" << std::setprecision(4) << last_val_loss
                  << " best_val_loss=" << best_val_loss << " best_val_acc=" << std::setprecision(2) << best_val_acc << std::setprecision(4)
                  << " monitor=" << monitor_name << " best_epoch=" << best_epoch
                  << " epochs_run=" << epochs_run << " stopped=" << (stopped_early ? 1 : 0)
                  << " time=" << std::setprecision(1) << total << "\n";
    } catch (const std::exception& e) {
        std::cerr << "HATA: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
