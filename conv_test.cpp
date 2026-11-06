#include "network.h"
#include "conv.h"
#include "dataloader.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "  [HATA] " << msg << "\n"; failures++; } } while (0)

// Parametrelere isaretci almak icin sahte optimizer (gradcheck_test.cpp ile ayni fikir)
struct Probe : Optimizer {
    std::map<std::string, Matrix*> params;
    void step(Matrix& p, const Matrix&, const std::string& id) override { params[id] = &p; }
    void set_learning_rate(double) override {}
    double learning_rate() const override { return 0.0; }
    std::string name() const override { return "Probe"; }
};

static double max_abs_diff(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e300;
    double d = 0; for (size_t i = 0; i < a.data.size(); i++) d = std::max(d, std::fabs(a.data[i] - b.data[i]));
    return d;
}

// ---- Naif referans: im2col KULLANMAYAN, dogrudan tanimdan konvolusyon ----
static Matrix ref_conv(const Matrix& x, const Matrix& W, const Matrix& bias,
                       int in_c, int in_h, int in_w, int out_c, int kh, int kw, int stride, int pad) {
    const int oh_n = (in_h + 2 * pad - kh) / stride + 1, ow_n = (in_w + 2 * pad - kw) / stride + 1;
    const int B = x.cols;
    Matrix out(out_c * oh_n * ow_n, B, 0.0);
    for (int b = 0; b < B; b++)
        for (int co = 0; co < out_c; co++)
            for (int oh = 0; oh < oh_n; oh++)
                for (int ow = 0; ow < ow_n; ow++) {
                    double s = bias.data[co];
                    for (int c = 0; c < in_c; c++)
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++) {
                                const int ih = oh * stride + ky - pad, iw = ow * stride + kx - pad;
                                if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;
                                s += W.at(co, (c * kh + ky) * kw + kx) * x.at((c * in_h + ih) * in_w + iw, b);
                            }
                    out.at((co * oh_n + oh) * ow_n + ow, b) = s;
                }
    return out;
}

static Matrix ref_pool(const Matrix& x, int ch, int in_h, int in_w, int pool, int stride) {
    const int oh_n = (in_h - pool) / stride + 1, ow_n = (in_w - pool) / stride + 1, B = x.cols;
    Matrix out(ch * oh_n * ow_n, B);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < ch; c++)
            for (int oh = 0; oh < oh_n; oh++)
                for (int ow = 0; ow < ow_n; ow++) {
                    double m = -1e300;
                    for (int dy = 0; dy < pool; dy++)
                        for (int dx = 0; dx < pool; dx++)
                            m = std::max(m, x.at((c * in_h + oh * stride + dy) * in_w + ow * stride + dx, b));
                    out.at((c * oh_n + oh) * ow_n + ow, b) = m;
                }
    return out;
}

int main() {
    Matrix::set_seed(77);

    // ------------------------------------------------------------
    std::cout << "=== 1) Conv2D ileri gecis: naif referansla karsilastirma ===\n";
    {
        struct Cfg { int c, h, w, oc, kh, kw, s, p, B; };
        const Cfg cfgs[] = {
            {1, 6, 6, 2, 3, 3, 1, 0, 4},      // temel
            {2, 7, 5, 3, 3, 3, 1, 1, 5},      // kare olmayan girdi + padding
            {3, 8, 8, 4, 2, 2, 2, 0, 3},      // stride 2
            {2, 9, 7, 3, 3, 2, 2, 1, 4},      // dikdortgen cekirdek + stride 2 + padding, tek sayili boyutlar
            {1, 5, 5, 2, 5, 5, 1, 0, 2},      // cekirdek = girdi (tek cikti piksel)
            {2, 4, 4, 2, 1, 1, 1, 0, 6},      // 1x1 konvolusyon
            {4, 6, 6, 5, 3, 3, 1, 1, 1},      // batch = 1
        };
        for (const auto& g : cfgs) {
            Conv2DLayer conv(g.c, g.h, g.w, g.oc, g.kh, g.kw, g.s, g.p);
            Probe probe;
            conv.update(&probe, "L");
            *probe.params["L_b"] = Matrix::random(g.oc, 1, 0.0, 0.8);       // sifirdan farkli bias
            Matrix x = Matrix::random(g.c * g.h * g.w, g.B);
            Matrix y = conv.forward(x);
            Matrix r = ref_conv(x, *probe.params["L_W"], *probe.params["L_b"], g.c, g.h, g.w, g.oc, g.kh, g.kw, g.s, g.p);
            const double d = max_abs_diff(y, r);
            std::ostringstream nm; nm << "c" << g.c << " " << g.h << "x" << g.w << " ->" << g.oc << " k" << g.kh << "x" << g.kw << " s" << g.s << " p" << g.p << " B" << g.B;
            std::cout << "  " << std::left << std::setw(34) << nm.str() << std::right << " cikti " << y.rows << "x" << y.cols
                      << "  max |fark| " << std::scientific << std::setprecision(2) << d << "\n";
            CHECK(d < 1e-12, "Conv2D ileri gecis referanstan sapiyor: " << nm.str());
            CHECK(y.rows == conv.out_features(), "out_features() cikti satir sayisiyla uyusmali");
        }
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 2) MaxPool2D ileri gecis: naif referansla karsilastirma ===\n";
    {
        struct Cfg { int c, h, w, k, s, B; };
        const Cfg cfgs[] = { {2, 6, 6, 2, 2, 4}, {3, 7, 5, 2, 2, 3}, {1, 6, 6, 3, 1, 2}, {2, 8, 8, 3, 2, 5}, {2, 4, 4, 4, 4, 2} };
        for (const auto& g : cfgs) {
            MaxPool2DLayer pool(g.c, g.h, g.w, g.k, g.s);
            Matrix x = Matrix::random(g.c * g.h * g.w, g.B);
            Matrix y = pool.forward(x);
            const double d = max_abs_diff(y, ref_pool(x, g.c, g.h, g.w, g.k, g.s));
            std::cout << "  c" << g.c << " " << g.h << "x" << g.w << " k" << g.k << " s" << g.s << " B" << g.B
                      << " -> " << y.rows << "x" << y.cols << "  max |fark| " << std::scientific << std::setprecision(2) << d << "\n";
            CHECK(d == 0.0, "MaxPool2D referanstan sapiyor");
            CHECK(y.rows == pool.out_features(), "out_features() uyusmali");
        }
        // backward: gradyan sadece kazanan elemana gitmeli ve toplam korunmali
        MaxPool2DLayer pool(2, 6, 6, 2, 2);
        Matrix x = Matrix::random(2 * 36, 3);
        Matrix y = pool.forward(x);
        Matrix g = Matrix::random(y.rows, y.cols);
        Matrix dx = pool.backward(g);
        double sg = 0, sdx = 0; int nz = 0;
        for (double v : g.data) sg += v;
        for (double v : dx.data) { sdx += v; nz += (v != 0.0); }
        CHECK(std::fabs(sg - sdx) < 1e-12, "backward gradyan toplamini korumali (cakismayan pencere)");
        CHECK(nz == (int)g.data.size(), "her cikti gradyani tam bir girdiye gitmeli (" << nz << " vs " << g.data.size() << ")");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 3) Oteleme (translation) ozelligi: girdi 1 piksel kayarsa cikti da kaymali ===\n";
    {
        Conv2DLayer conv(1, 10, 10, 3, 3, 3, 1, 1);          // 'same' padding
        Matrix a(100, 1, 0.0), bsh(100, 1, 0.0);
        std::mt19937 rng(5); std::uniform_real_distribution<double> u(-1, 1);
        for (int h = 3; h < 6; h++) for (int w = 2; w < 5; w++) { double v = u(rng); a.at(h * 10 + w, 0) = v; bsh.at(h * 10 + (w + 1), 0) = v; }   // blob 1 sutun saga
        Matrix ya = conv.forward(a), yb = conv.forward(bsh);
        // bias=0 oldugu icin kenar etkisi olmayan bolgede yb(h, w+1) == ya(h, w)
        double worst = 0;
        for (int c = 0; c < 3; c++) for (int h = 1; h < 9; h++) for (int w = 1; w < 8; w++)
            worst = std::max(worst, std::fabs(yb.at((c * 10 + h) * 10 + (w + 1), 0) - ya.at((c * 10 + h) * 10 + w, 0)));
        std::cout << "  max |fark| (ic bolge) " << std::scientific << std::setprecision(2) << worst << "\n";
        CHECK(worst < 1e-12, "oteleme esdegiskenligi bozuk");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 4) Gecersiz parametreler ve boyut kontrolleri ===\n";
    {
        auto throws = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        CHECK(throws([]{ Conv2DLayer c(1, 5, 5, 2, 7); }),                 "cekirdek girdiden buyuk");
        CHECK(throws([]{ Conv2DLayer c(1, 5, 5, 2, 3, 0, 0); }),           "stride = 0");
        CHECK(throws([]{ Conv2DLayer c(1, 5, 5, 2, 3, 1, -1); }),          "pad < 0");
        CHECK(throws([]{ Conv2DLayer c(0, 5, 5, 2, 3); }),                 "kanal = 0");
        CHECK(throws([]{ MaxPool2DLayer p(1, 3, 3, 4); }),                 "pencere girdiden buyuk");
        CHECK(throws([]{ MaxPool2DLayer p(1, 4, 4, 0); }),                 "pool = 0");
        Conv2DLayer conv(2, 5, 5, 3, 3);
        CHECK(throws([&]{ conv.forward(Matrix(49, 2)); }),                 "yanlis girdi boyutu (Conv2D)");
        MaxPool2DLayer pool(2, 4, 4, 2);
        CHECK(throws([&]{ pool.forward(Matrix(31, 2)); }),                 "yanlis girdi boyutu (MaxPool2D)");
        Conv2DLayer c1(1, 28, 28, 8, 3, 1, 1);
        std::cout << "  " << c1.name() << " -> " << c1.out_channels() << "x" << c1.out_height() << "x" << c1.out_width()
                  << " (" << c1.out_features() << " ozellik)\n";
        CHECK(c1.out_height() == 28 && c1.out_width() == 28, "'same' padding boyutu korumali");
        MaxPool2DLayer p1(8, 28, 28, 2);
        CHECK(p1.out_height() == 14 && p1.out_features() == 8 * 14 * 14, "pool 2 boyutu yariya indirmeli");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 5) Kaydet / yukle / snapshot (Conv2D agirliklari dahil) ===\n";
    {
        auto build = [](unsigned seed) {
            Matrix::set_seed(seed);
            NeuralNetwork net(std::make_shared<Adam>(0.01));
            net.add(std::make_unique<Conv2DLayer>(1, 8, 8, 3, 3, 1, 1));
            net.add(std::make_unique<ReLULayer>());
            net.add(std::make_unique<MaxPool2DLayer>(3, 8, 8, 2));
            net.add(std::make_unique<DenseLayer>(3 * 4 * 4, 2));
            net.add(std::make_unique<SoftmaxLayer>());
            return net;
        };
        NeuralNetwork net = build(1);
        Matrix X = Matrix::random(64, 10);
        Matrix Y = one_hot({0, 1, 0, 1, 1, 0, 0, 1, 1, 0}, 2);
        for (int i = 0; i < 15; i++) net.train_step(X, Y, "ce");
        const std::string snap = net.snapshot();
        net.set_training(false); Matrix p1 = net.predict(X);
        for (int i = 0; i < 25; i++) net.train_step(X, Y, "ce");
        net.restore(snap);
        Matrix p2 = net.predict(X);
        CHECK(max_abs_diff(p1, p2) == 0.0, "restore sonrasi tahminler bit-bit ayni olmali");

        NeuralNetwork other = build(99);                       // farkli rastgele agirliklar, ayni mimari
        other.restore(snap);
        other.set_training(false);
        CHECK(max_abs_diff(other.predict(X), p1) == 0.0, "ayni mimariye yukleme ayni tahmini vermeli");

        Matrix::set_seed(1);
        NeuralNetwork diff(std::make_shared<Adam>());
        diff.add(std::make_unique<Conv2DLayer>(1, 8, 8, 3, 5, 1, 1));    // k5: farkli
        diff.add(std::make_unique<ReLULayer>());
        diff.add(std::make_unique<MaxPool2DLayer>(3, 6, 6, 2));
        diff.add(std::make_unique<DenseLayer>(3 * 3 * 3, 2));
        diff.add(std::make_unique<SoftmaxLayer>());
        bool threw = false;
        try { diff.restore(snap); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw, "farkli cekirdek boyutuna yukleme hata vermeli");
        std::cout << "  weight_norm (conv + dense) = " << std::fixed << std::setprecision(3) << net.weight_norm() << "\n";
        CHECK(net.weight_norm() > 0.0, "weight_norm Conv2D agirliklarini da saymali");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 6) Ogrenme testi: yatay mi dikey mi cizgi? (8x8 sentetik goruntu, gurultulu) ===\n";
    {
        auto make = [](int n, unsigned seed, Matrix& X, Matrix& Y) {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> pos(1, 6);
            std::normal_distribution<double> noise(0.0, 0.3);
            X = Matrix(64, n); std::vector<int> lab(n);
            for (int j = 0; j < n; j++) {
                lab[j] = j % 2;                                   // 0: yatay, 1: dikey
                const int p = pos(rng);
                for (int i = 0; i < 64; i++) X.at(i, j) = noise(rng);
                for (int t = 0; t < 8; t++) X.at(lab[j] == 0 ? p * 8 + t : t * 8 + p, j) += 1.0;
            }
            Y = one_hot(lab, 2);
        };
        Matrix Xtr(1, 1), Ytr(1, 1), Xva(1, 1), Yva(1, 1);
        make(200, 1, Xtr, Ytr); make(200, 2, Xva, Yva);

        Matrix::set_seed(3);
        NeuralNetwork net(std::make_shared<Adam>(0.01));
        net.add(std::make_unique<Conv2DLayer>(1, 8, 8, 4, 3, 1, 1));
        net.add(std::make_unique<ReLULayer>());
        net.add(std::make_unique<MaxPool2DLayer>(4, 8, 8, 2));
        net.add(std::make_unique<DenseLayer>(4 * 4 * 4, 2));
        net.add(std::make_unique<SoftmaxLayer>());
        DataLoader train(Xtr, Ytr, 20, true, false, 4), val(Xva, Yva, 100, false);
        const double before = net.accuracy(val);
        net.fit(train, 25, "ce", 0, &val);
        const double after = net.accuracy(val);
        std::cout << "  val dogrulugu: " << std::fixed << std::setprecision(3) << before << " -> " << after << "\n";
        CHECK(after > 0.95, "CNN bu basit gorevi ogrenmeli (dogruluk " << after << ")");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n" << (failures == 0 ? "TUM TESTLER GECTI" : "BASARISIZ KONTROL SAYISI: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
