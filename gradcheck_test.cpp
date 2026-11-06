#include "network.h"
#include "conv.h"
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>

// ============================================================
//  Gradyan kontrol testleri
//
//  Analitik gradyani (backward) sayisal turevle (merkezi fark) karsilastirir:
//      dL/dw  ~  ( L(w+h) - L(w-h) ) / (2h)
//  Bu, backprop'taki turev hatalarini yakalamanin en guvenilir yoludur.
//
//  Parametre gradyanlarina katmanlarin icini degistirmeden ulasmak icin
//  Layer::update(Optimizer*, id) arayuzunu kullaniriz: GradProbe, step()
//  cagrisinda parametreye bir isaretci ve gradyanin kopyasini saklar.
// ============================================================

struct GradProbe : Optimizer {
    std::map<std::string, Matrix*> params;
    std::map<std::string, Matrix>  grads;
    void step(Matrix& p, const Matrix& g, const std::string& id) override {
        params[id] = &p;
        grads.erase(id);
        grads.emplace(id, g);
    }
    void set_learning_rate(double) override {}
    double learning_rate() const override { return 0.0; }
    std::string name() const override { return "GradProbe"; }
};

static int failures = 0;
static const double H   = 1e-5;   // merkezi fark adimi
// allclose tarzi tolerans: |analitik - sayisal| <= ATOL + RTOL * max(|analitik|, |sayisal|)
//  - RTOL: gradyan buyukken bagil dogruluk (1e-6)
//  - ATOL: gradyan ~0 iken sayisal turevin yuvarlama gurultusu (~1e-11) hata sayilmasin
//    (ornegin BatchNorm'dan once gelen bias'in gradyani matematiksel olarak 0'dir)
static const double ATOL = 1e-8;
static const double RTOL = 1e-6;

template <class F>
static Matrix numeric_grad(Matrix& x, F loss) {
    Matrix g(x.rows, x.cols);
    for (size_t i = 0; i < x.data.size(); i++) {
        const double old = x.data[i];
        x.data[i] = old + H; const double lp = loss();
        x.data[i] = old - H; const double lm = loss();
        x.data[i] = old;
        g.data[i] = (lp - lm) / (2 * H);
    }
    return g;
}

// En kotu elemanin "tolerans birimi" cinsinden hatasi: <= 1 ise gecer
static double worst_score(const Matrix& a, const Matrix& n, double* max_abs = nullptr) {
    double worst = 0.0, mabs = 0.0;
    for (size_t i = 0; i < a.data.size(); i++) {
        const double d = std::fabs(a.data[i] - n.data[i]);
        const double tol = ATOL + RTOL * std::max(std::fabs(a.data[i]), std::fabs(n.data[i]));
        worst = std::max(worst, d / tol);
        mabs  = std::max(mabs, d);
    }
    if (max_abs) *max_abs = mabs;
    return worst;
}

// analitik/sayisal norm orani (hata varsa olcek hatasini ele verir)
static double norm_ratio(const Matrix& a, const Matrix& n) {
    double na = 0, nn = 0;
    for (size_t i = 0; i < a.data.size(); i++) { na += a.data[i] * a.data[i]; nn += n.data[i] * n.data[i]; }
    return nn > 0 ? std::sqrt(na / nn) : 0.0;
}

static bool report(const std::string& name, const Matrix& analytic, const Matrix& numeric, bool expect_fail = false) {
    double mabs = 0;
    const double score = worst_score(analytic, numeric, &mabs);
    const bool ok = score <= 1.0;
    const bool good = expect_fail ? !ok : ok;
    std::cout << "  " << std::left << std::setw(40) << name << std::right
              << "  max |fark| " << std::scientific << std::setprecision(2) << mabs
              << (good ? "   OK" : "   HATA");
    if (!ok) std::cout << "   (analitik/sayisal norm orani = " << std::fixed << std::setprecision(4) << norm_ratio(analytic, numeric) << ")";
    std::cout << "\n";
    if (!good) failures++;
    return good;
}

static Matrix rnd(int r, int c, double sd = 1.0) { return Matrix::random(r, c, 0.0, sd); }
static double dot_sum(const Matrix& a, const Matrix& b) {   // sum(a ⊙ b)
    double s = 0; for (size_t i = 0; i < a.data.size(); i++) s += a.data[i] * b.data[i]; return s;
}

// ---- Tek katman kontrolu: L = sum(R ⊙ forward(x)) ----
static void check_layer(const std::string& name, Layer& layer, Matrix x, const std::vector<std::string>& param_ids,
                        bool expect_fail = false, bool check_dx = true) {
    GradProbe probe;
    Matrix y = layer.forward(x);
    Matrix R = rnd(y.rows, y.cols);                 // rastgele "yukaridan gelen" gradyan

    // Parametreli katmanlarda parametreleri varsayilandan uzaklastir (gamma=1, beta=0 hatalari gizlemesin)
    if (!param_ids.empty()) {
        layer.update(&probe, "L");
        for (const auto& id : param_ids)
            if (id.find("gamma") != std::string::npos || id.find("beta") != std::string::npos)
                *probe.params["L_" + id] = rnd(probe.params["L_" + id]->rows, probe.params["L_" + id]->cols, 0.7);
        y = layer.forward(x);
    }

    Matrix dx = layer.backward(R);
    layer.update(&probe, "L");

    auto loss = [&]() { return dot_sum(R, layer.forward(x)); };
    if (check_dx) report(name + " : dx", dx, numeric_grad(x, loss), expect_fail);
    for (const auto& id : param_ids)
        report(name + " : " + id, probe.grads.at("L_" + id), numeric_grad(*probe.params["L_" + id], loss), expect_fail);
}

// ---- Kasitli olarak bozuk katman: negatif kontrol ----
class BrokenTanh : public Layer {
    Matrix y;
public:
    BrokenTanh() : y(1, 1) {}
    Matrix forward(const Matrix& x) override { y = x.apply([](double v){ return std::tanh(v); }); return y; }
    Matrix backward(const Matrix& g) override {
        // YANLIS: turev (1 - y^2) olmali, burada (1 - y)
        return g.hadamard(y.apply([](double v){ return 1.0 - v; }));
    }
    std::string name() const override { return "BrokenTanh"; }
};

// ---- Tum ag kontrolu ----
using Builder = std::function<void(NeuralNetwork&)>;

static double net_loss(NeuralNetwork& net, const Matrix& X, const Matrix& Y, const std::string& type) {
    Matrix p = net.predict(X);
    if (type == "ce")  return CrossEntropyLoss::compute(p, Y);
    if (type == "bce") return BCELoss::compute(p, Y);
    return MSELoss::compute(p, Y);
}

static void check_network(const std::string& name, const Builder& build, const Matrix& X, const Matrix& Y,
                          const std::string& loss_type, unsigned seed) {
    auto probe = std::make_shared<GradProbe>();
    Matrix::set_seed(seed);
    NeuralNetwork net(probe);
    build(net);
    net.train_step(X, Y, loss_type);                // ileri + geri; probe gradyanlari kaydeder
    std::cout << "  [" << name << "]\n";
    for (auto& kv : probe->params) {
        auto loss = [&]() { return net_loss(net, X, Y, loss_type); };
        report("  " + kv.first, probe->grads.at(kv.first), numeric_grad(*kv.second, loss));
    }
}

int main() {
    Matrix::set_seed(2024);

    // ------------------------------------------------------------
    std::cout << "=== 1) Aktivasyon katmanlari (girdi gradyani) ===\n";
    {
        Matrix x = rnd(4, 6);
        for (auto& v : x.data) if (std::fabs(v) < 0.05) v = 0.5;    // ReLU kirilma noktasindan uzak dur
        ReLULayer relu;       check_layer("ReLU",    relu,    x, {});
        SigmoidLayer sig;     check_layer("Sigmoid", sig,     x, {});
        TanhLayer th;         check_layer("Tanh",    th,      x, {});
        SoftmaxLayer sm;      check_layer("Softmax", sm,      x, {});
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 2) DenseLayer (dx, dW, db) ===\n";
    {
        DenseLayer d(5, 3);
        check_layer("Dense(5->3), batch 6", d, rnd(5, 6), {"W", "b"});
        DenseLayer d1(4, 4);
        check_layer("Dense(4->4), batch 1", d1, rnd(4, 1), {"W", "b"});

        // Ilk katman modu: dx hesaplanmaz ama dW/db dogru kalmali
        DenseLayer first(5, 3);
        first.set_compute_input_grad(false);
        Matrix x = rnd(5, 6);
        GradProbe probe;
        Matrix y = first.forward(x);
        Matrix R = rnd(y.rows, y.cols);
        Matrix dx = first.backward(R);
        first.update(&probe, "L");
        auto loss = [&]() { return dot_sum(R, first.forward(x)); };
        report("Dense (dx kapali) : dW", probe.grads.at("L_W"), numeric_grad(*probe.params["L_W"], loss));
        report("Dense (dx kapali) : db", probe.grads.at("L_b"), numeric_grad(*probe.params["L_b"], loss));
        if (dx.rows != 1 || dx.cols != 1) { std::cout << "  [HATA] dx kapaliyken 1x1 donmeli\n"; failures++; }
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 3) BatchNormLayer (dx, dgamma, dbeta) ===\n";
    {
        BatchNormLayer bn(4);
        check_layer("BatchNorm(4), batch 6", bn, rnd(4, 6, 1.5), {"gamma", "beta"});
        BatchNormLayer bn2(3);
        check_layer("BatchNorm(3), batch 12", bn2, rnd(3, 12), {"gamma", "beta"});
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 4) DropoutLayer (rastgele: maske iliskisi ve istatistik) ===\n";
    {
        const double p = 0.3;
        DropoutLayer drop(p, 11);
        Matrix ones = Matrix::ones(200, 200);
        Matrix mask = drop.forward(ones);          // girdi 1 oldugu icin cikti = maskenin kendisi
        int zeros = 0; bool values_ok = true; double sum = 0;
        for (double v : mask.data) {
            if (v == 0.0) zeros++;
            else if (std::fabs(v - 1.0 / (1.0 - p)) > 1e-12) values_ok = false;
            sum += v;
        }
        const double zfrac = (double)zeros / mask.data.size(), mean = sum / mask.data.size();
        std::cout << std::fixed << std::setprecision(4) << "  sifirlanan oran " << zfrac << " (beklenen " << p
                  << "), ortalama " << mean << " (beklenen 1)\n";
        if (!values_ok)                          { std::cout << "  [HATA] maske degerleri {0, 1/(1-p)} disinda\n"; failures++; }
        if (std::fabs(zfrac - p) > 0.01)         { std::cout << "  [HATA] sifirlama orani\n"; failures++; }
        if (std::fabs(mean - 1.0) > 0.02)        { std::cout << "  [HATA] ters (inverted) olcekleme: ortalama 1 olmali\n"; failures++; }

        Matrix R = rnd(200, 200);
        Matrix g = drop.backward(R);
        report("Dropout : backward = R ⊙ maske", g, R.hadamard(mask));

        drop.set_training(false);
        Matrix x = rnd(5, 5);
        Matrix y = drop.forward(x);
        bool same = true; for (size_t i = 0; i < x.data.size(); i++) same &= (x.data[i] == y.data[i]);
        Matrix gR = rnd(5, 5);
        Matrix gb = drop.backward(gR);
        for (size_t i = 0; i < gR.data.size(); i++) same &= (gR.data[i] == gb.data[i]);
        if (!same) { std::cout << "  [HATA] eval modunda girdi/gradyan aynen gecmeli\n"; failures++; }
        else std::cout << "  eval modu: girdi ve gradyan aynen geciyor   OK\n";

        DropoutLayer a(p, 5), b(p, 5);
        Matrix ma = a.forward(ones), mb = b.forward(ones);
        bool det = true; for (size_t i = 0; i < ma.data.size(); i++) det &= (ma.data[i] == mb.data[i]);
        if (!det) { std::cout << "  [HATA] ayni tohum ayni maskeyi uretmeli\n"; failures++; }
        else std::cout << "  ayni tohum -> ayni maske                    OK\n";
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 5) Kayip fonksiyonlari (tahmine gore gradyan) ===\n";
    {
        Matrix pred = rnd(3, 5), target = rnd(3, 5);
        report("MSE", MSELoss::gradient(pred, target),
               numeric_grad(pred, [&]{ return MSELoss::compute(pred, target); }));

        Matrix p01(2, 6), t01(2, 6);
        for (size_t i = 0; i < p01.data.size(); i++) { p01.data[i] = 0.1 + 0.8 * (0.5 + 0.5 * std::sin(1.7 * i + 0.3)); t01.data[i] = (i % 3 == 0) ? 1.0 : 0.0; }
        report("BCE", BCELoss::gradient(p01, t01),
               numeric_grad(p01, [&]{ return BCELoss::compute(p01, t01); }));

        Matrix probs(4, 5);
        for (int j = 0; j < 5; j++) {                       // her sutun bir olasilik dagilimi
            double s = 0; for (int i = 0; i < 4; i++) { probs.at(i, j) = 0.2 + std::fabs(std::sin(1.3 * (i + 1) * (j + 1))); s += probs.at(i, j); }
            for (int i = 0; i < 4; i++) probs.at(i, j) /= s;
        }
        Matrix onehot = one_hot({0, 3, 1, 2, 3}, 4);
        report("CrossEntropy", CrossEntropyLoss::gradient(probs, onehot),
               numeric_grad(probs, [&]{ return CrossEntropyLoss::compute(probs, onehot); }));
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 6) Tum ag (ileri + geri + kayip, her parametre) ===\n";
    {
        Matrix X = rnd(3, 8);
        Matrix Yb(1, 8), Ym = rnd(2, 8);
        for (int j = 0; j < 8; j++) Yb.at(0, j) = (j % 2);
        Matrix Yc = one_hot({0, 1, 2, 1, 0, 2, 2, 1}, 3);

        check_network("Dense-Tanh-Dense-Sigmoid + BCE", [](NeuralNetwork& n) {
            n.add(std::make_unique<DenseLayer>(3, 5)); n.add(std::make_unique<TanhLayer>());
            n.add(std::make_unique<DenseLayer>(5, 1)); n.add(std::make_unique<SigmoidLayer>());
        }, X, Yb, "bce", 1);

        check_network("Dense-Tanh-Dense + MSE", [](NeuralNetwork& n) {
            n.add(std::make_unique<DenseLayer>(3, 4)); n.add(std::make_unique<TanhLayer>());
            n.add(std::make_unique<DenseLayer>(4, 2));
        }, X, Ym, "mse", 2);

        check_network("Dense-BN-Tanh-Dense-Softmax + CE", [](NeuralNetwork& n) {
            n.add(std::make_unique<DenseLayer>(3, 6));
            n.add(std::make_unique<BatchNormLayer>(6));
            n.add(std::make_unique<TanhLayer>());
            n.add(std::make_unique<DenseLayer>(6, 3));
            n.add(std::make_unique<SoftmaxLayer>());
        }, X, Yc, "ce", 3);

        check_network("Dense-ReLU-Dense-Softmax + CE", [](NeuralNetwork& n) {
            n.add(std::make_unique<DenseLayer>(3, 7)); n.add(std::make_unique<ReLULayer>());
            n.add(std::make_unique<DenseLayer>(7, 3)); n.add(std::make_unique<SoftmaxLayer>());
        }, X, Yc, "ce", 4);
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 7) Batch boyutu degismezligi: ayni veriyi iki kez vermek ayni ortalama gradyani vermeli ===\n";
    {
        Matrix X = rnd(3, 4), Y(1, 4, {0, 1, 1, 0});
        Matrix X2(3, 8), Y2(1, 8);
        for (int j = 0; j < 8; j++) {
            for (int i = 0; i < 3; i++) X2.at(i, j) = X.at(i, j % 4);
            Y2.at(0, j) = Y.at(0, j % 4);
        }
        auto grads_of = [&](const Matrix& x, const Matrix& y) {
            auto probe = std::make_shared<GradProbe>();
            Matrix::set_seed(9);
            NeuralNetwork net(probe);
            net.add(std::make_unique<DenseLayer>(3, 5)); net.add(std::make_unique<TanhLayer>());
            net.add(std::make_unique<DenseLayer>(5, 1)); net.add(std::make_unique<SigmoidLayer>());
            net.train_step(x, y, "bce");
            return probe;
        };
        auto a = grads_of(X, Y), b = grads_of(X2, Y2);
        for (auto& kv : a->grads)
            report("layer " + kv.first + " (4 vs 8 ornek)", kv.second, b->grads.at(kv.first));
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 8) Negatif kontrol: kasitli bozuk katman YAKALANMALI ===\n";
    {
        BrokenTanh broken;
        check_layer("BrokenTanh (bekleniyor: HATA bulunur)", broken, rnd(4, 6), {}, /*expect_fail=*/true);
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 9) Conv2DLayer (dx, dW, db): stride / padding / dikdortgen cekirdek / cok kanal ===\n";
    {
        struct Cfg { int c, h, w, oc, kh, kw, s, p, B; };
        const Cfg cfgs[] = {
            {1, 5, 5, 2, 3, 3, 1, 0, 3},
            {2, 6, 5, 3, 3, 3, 1, 1, 4},       // kare olmayan girdi + padding
            {2, 7, 7, 2, 3, 3, 2, 1, 2},       // stride 2
            {3, 6, 7, 2, 2, 3, 1, 0, 3},       // dikdortgen cekirdek (kh != kw)
            {2, 8, 6, 3, 3, 2, 2, 1, 2},       // dikdortgen + stride 2 + padding
            {1, 4, 4, 2, 1, 1, 1, 0, 3},       // 1x1
            {2, 5, 5, 2, 3, 3, 1, 1, 1},       // batch = 1
        };
        for (const auto& g : cfgs) {
            Conv2DLayer conv(g.c, g.h, g.w, g.oc, g.kh, g.kw, g.s, g.p);
            std::ostringstream nm;
            nm << "Conv c" << g.c << " " << g.h << "x" << g.w << "->" << g.oc << " k" << g.kh << "x" << g.kw << " s" << g.s << " p" << g.p << " B" << g.B;
            check_layer(nm.str(), conv, rnd(g.c * g.h * g.w, g.B), {"W", "b"});
        }

        // ilk katman modu: dx hesaplanmaz ama dW/db dogru kalmali
        Conv2DLayer first(2, 6, 6, 3, 3, 1, 1);
        first.set_compute_input_grad(false);
        Matrix x = rnd(2 * 36, 3);
        GradProbe probe;
        Matrix y = first.forward(x);
        Matrix R = rnd(y.rows, y.cols);
        Matrix dx = first.backward(R);
        first.update(&probe, "L");
        auto loss = [&]() { return dot_sum(R, first.forward(x)); };
        report("Conv (dx kapali) : dW", probe.grads.at("L_W"), numeric_grad(*probe.params["L_W"], loss));
        report("Conv (dx kapali) : db", probe.grads.at("L_b"), numeric_grad(*probe.params["L_b"], loss));
        if (dx.rows != 1 || dx.cols != 1) { std::cout << "  [HATA] dx kapaliyken 1x1 donmeli\n"; failures++; }
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 10) MaxPool2DLayer (girdi gradyani) ===\n";
    {
        MaxPool2DLayer p1(2, 6, 6, 2, 2);   check_layer("MaxPool c2 6x6 k2 s2", p1, rnd(2 * 36, 3), {});
        MaxPool2DLayer p2(1, 7, 5, 3, 2);   check_layer("MaxPool c1 7x5 k3 s2 (cakisan, kalan satir/sutun)", p2, rnd(35, 2), {});
        MaxPool2DLayer p3(3, 5, 5, 2, 1);   check_layer("MaxPool c3 5x5 k2 s1 (cakisan pencereler)", p3, rnd(75, 4), {});
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 11) Tam CNN (ileri + geri + kayip, her parametre) ===\n";
    {
        Matrix X = rnd(1 * 6 * 6, 4);
        Matrix Yc = one_hot({0, 2, 1, 2}, 3);

        check_network("Conv-Tanh-Pool-Dense-Softmax + CE", [](NeuralNetwork& n) {
            n.add(std::make_unique<Conv2DLayer>(1, 6, 6, 2, 3, 1, 1));       // 2x6x6
            n.add(std::make_unique<TanhLayer>());
            n.add(std::make_unique<MaxPool2DLayer>(2, 6, 6, 2));             // 2x3x3
            n.add(std::make_unique<DenseLayer>(2 * 3 * 3, 3));
            n.add(std::make_unique<SoftmaxLayer>());
        }, X, Yc, "ce", 41);

        check_network("Conv-Tanh-Pool-Conv-Tanh-Dense-Softmax + CE (iki conv)", [](NeuralNetwork& n) {
            n.add(std::make_unique<Conv2DLayer>(1, 6, 6, 2, 3, 1, 1));       // 2x6x6
            n.add(std::make_unique<TanhLayer>());
            n.add(std::make_unique<MaxPool2DLayer>(2, 6, 6, 2));             // 2x3x3
            n.add(std::make_unique<Conv2DLayer>(2, 3, 3, 3, 2, 2, 1, 0));       // 3x2x2 (dikdortgen kurucu)
            n.add(std::make_unique<TanhLayer>());
            n.add(std::make_unique<DenseLayer>(3 * 2 * 2, 3));
            n.add(std::make_unique<SoftmaxLayer>());
        }, X, Yc, "ce", 42);
    }

    // ------------------------------------------------------------
    std::cout << "\n=== 12) CNN icin batch boyutu degismezligi (4 ornek vs ayni 4 ornek iki kez) ===\n";
    {
        Matrix X = rnd(36, 4), Y = one_hot({0, 1, 1, 0}, 2);
        Matrix X2(36, 8), Y2(2, 8);
        for (int j = 0; j < 8; j++) {
            for (int i = 0; i < 36; i++) X2.at(i, j) = X.at(i, j % 4);
            for (int i = 0; i < 2; i++)  Y2.at(i, j) = Y.at(i, j % 4);
        }
        auto grads_of = [&](const Matrix& x, const Matrix& y) {
            auto probe = std::make_shared<GradProbe>();
            Matrix::set_seed(17);
            NeuralNetwork net(probe);
            net.add(std::make_unique<Conv2DLayer>(1, 6, 6, 2, 3, 1, 1));
            net.add(std::make_unique<TanhLayer>());
            net.add(std::make_unique<MaxPool2DLayer>(2, 6, 6, 2));
            net.add(std::make_unique<DenseLayer>(18, 2));
            net.add(std::make_unique<SoftmaxLayer>());
            net.train_step(x, y, "ce");
            return probe;
        };
        auto a = grads_of(X, Y), b = grads_of(X2, Y2);
        for (auto& kv : a->grads)
            report("katman " + kv.first + " (4 vs 8 ornek)", kv.second, b->grads.at(kv.first));
    }

    std::cout << "\n" << (failures == 0 ? "TUM TESTLER GECTI" : "BASARISIZ KONTROL SAYISI: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
