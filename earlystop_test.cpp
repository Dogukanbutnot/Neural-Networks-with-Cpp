#include "network.h"
#include "earlystopping.h"
#include "dataloader.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "  [HATA] " << msg << "\n"; failures++; } } while (0)
static bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

// ---------- Oyuncak problem: etiketi gurultulu kucuk egitim seti -> ezberleme ----------
struct Problem {
    Matrix Xtr, Ytr, Xva, Yva;
    Problem() : Xtr(1, 1), Ytr(1, 1), Xva(1, 1), Yva(1, 1) {}
};

static Problem make_problem(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    auto gen = [&](int n, double flip, Matrix& X, Matrix& Y) {
        X = Matrix(2, n);
        std::vector<int> lab(n);
        for (int j = 0; j < n; j++) {
            X.at(0, j) = u(rng); X.at(1, j) = u(rng);
            int l = (X.at(0, j) * X.at(1, j) > 0) ? 1 : 0;      // dogrusal olmayan: ceyrek bolgeler
            if (coin(rng) < flip) l = 1 - l;                     // etiket gurultusu
            lab[j] = l;
        }
        Y = one_hot(lab, 2);
    };
    Problem p;
    gen(60,  0.25, p.Xtr, p.Ytr);    // kucuk + %25 gurultulu: ezberlemeye cok aciktir
    gen(300, 0.05, p.Xva, p.Yva);
    return p;
}

static NeuralNetwork make_net(unsigned seed) {
    Matrix::set_seed(seed);
    NeuralNetwork net(std::make_shared<Adam>(0.01));
    net.add(std::make_unique<DenseLayer>(2, 48));
    net.add(std::make_unique<BatchNormLayer>(48));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DropoutLayer>(0.1, 3));            // tohumlu: tekrarlanabilir
    net.add(std::make_unique<DenseLayer>(48, 48));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(48, 2));
    net.add(std::make_unique<SoftmaxLayer>());
    return net;
}

int main() {
    // ------------------------------------------------------------
    std::cout << "=== 1) Durdurma mantigi (yapay metrik dizileriyle) ===\n";
    {
        EarlyStopping es(2);                                   // patience 2
        const double m[] = {1.0, 0.9, 0.95, 0.92};
        const bool imp[] = {true, true, false, false};
        for (int i = 0; i < 4; i++) CHECK(es.on_epoch_end(m[i]) == imp[i], "a) iyilesme bayragi epoch " << i);
        CHECK(es.should_stop() && es.stopped_epoch() == 3 && es.best_epoch() == 1 && near(es.best_value(), 0.9),
              "a) epoch 3'te durmali, en iyi epoch 1 (0.9)");

        EarlyStopping e2(2);
        for (double v : {1.0, 1.1, 0.9, 1.0}) e2.on_epoch_end(v);
        CHECK(!e2.should_stop(), "b) iyilesme beklemeyi sifirlamali (henuz durmamali)");
        e2.on_epoch_end(1.0);
        CHECK(e2.should_stop() && e2.stopped_epoch() == 4 && e2.best_epoch() == 2, "b) epoch 4'te durmali, en iyi epoch 2");

        EarlyStopping e3(2, /*min_delta=*/0.1);                // 0.05'lik iyilesme sayilmaz
        for (double v : {1.0, 0.95, 0.92}) e3.on_epoch_end(v);
        CHECK(e3.should_stop() && e3.best_epoch() == 0 && near(e3.best_value(), 1.0), "c) min_delta: kucuk iyilesmeler sayilmamali");
        EarlyStopping e3b(2, 0.1);
        for (double v : {1.0, 0.95, 0.85}) e3b.on_epoch_end(v);
        CHECK(!e3b.should_stop() && e3b.best_epoch() == 2, "c) min_delta'dan buyuk iyilesme sayilmali");

        const double nan = std::numeric_limits<double>::quiet_NaN();
        EarlyStopping e4(2);
        for (double v : {1.0, nan, nan}) e4.on_epoch_end(v);
        CHECK(e4.should_stop() && e4.best_epoch() == 0 && near(e4.best_value(), 1.0), "d) NaN iyilesme sayilmamali");
        EarlyStopping e5(1);
        e5.on_epoch_end(nan);
        CHECK(e5.best_epoch() == -1 && e5.should_stop(), "e) ilk metrik NaN: en iyi epoch yok");

        bool t1 = false, t2 = false;
        try { EarlyStopping bad(0); } catch (const std::invalid_argument&) { t1 = true; }
        try { EarlyStopping bad(2, -0.1); } catch (const std::invalid_argument&) { t2 = true; }
        CHECK(t1 && t2, "f) gecersiz parametreler hata vermeli");

        es.reset();
        CHECK(!es.should_stop() && es.epochs_seen() == 0 && es.best_epoch() == -1 && es.history().empty(), "g) reset()");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 2) Snapshot / restore: bit-bit geri donus (BatchNorm istatistikleri dahil) ===\n";
    {
        Problem pr = make_problem(1);
        NeuralNetwork net = make_net(11);
        for (int i = 0; i < 20; i++) net.train_step(pr.Xtr, pr.Ytr, "ce");
        const std::string s1 = net.snapshot();
        net.set_training(false);
        Matrix p1 = net.predict(pr.Xva);

        for (int i = 0; i < 40; i++) net.train_step(pr.Xtr, pr.Ytr, "ce");    // agirliklar + BN running stats degisir
        net.set_training(false);
        Matrix p2 = net.predict(pr.Xva);
        bool differs = false;
        for (size_t i = 0; i < p1.data.size(); i++) differs |= (p1.data[i] != p2.data[i]);
        CHECK(differs, "saglik kontrolu: egitim tahminleri degistirmeli");

        net.restore(s1);
        Matrix p3 = net.predict(pr.Xva);
        bool same = true;
        for (size_t i = 0; i < p1.data.size(); i++) same &= (p1.data[i] == p3.data[i]);
        CHECK(same, "restore sonrasi tahminler bit-bit ayni olmali");
        CHECK(net.snapshot() == s1, "restore sonrasi snapshot bayt bayt ayni olmali");

        NeuralNetwork other(std::make_shared<Adam>());
        other.add(std::make_unique<DenseLayer>(2, 5));
        bool threw = false;
        try { other.restore(s1); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw, "farkli mimariye restore hata vermeli");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 3) Ag entegrasyonu: ezberleyen problemde gercekten duruyor mu? ===\n";
    const int MAXE = 150, PAT = 8;
    {
        Problem pr = make_problem(2);
        NeuralNetwork net = make_net(21);
        DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
        auto es = std::make_shared<EarlyStopping>(PAT, 0.0, /*restore_best=*/true);
        net.set_early_stopping(es);
        net.fit(train, MAXE, "ce", 0, &val);

        const auto& h = es->history();
        double mn = h[0]; int mn_i = 0;
        for (size_t i = 1; i < h.size(); i++) if (h[i] < mn) { mn = h[i]; mn_i = (int)i; }
        std::cout << "  " << h.size() << " / " << MAXE << " epoch kosuldu; en iyi epoch " << es->best_epoch() + 1
                  << ", val loss " << std::fixed << std::setprecision(4) << es->best_value()
                  << " (durdugu epoch'taki val loss " << h.back() << ")\n";
        CHECK(es->should_stop() && (int)h.size() < MAXE, "ezberleyen problemde erken durmali");
        CHECK(near(es->best_value(), mn) && es->best_epoch() == mn_i, "best_value/best_epoch = gecmisin minimumu olmali");
        CHECK(es->stopped_epoch() - es->best_epoch() == PAT, "durma = en iyiden tam patience epoch sonra");
        CHECK(h.back() > es->best_value(), "durdugu epoch en iyiden kotu olmali");
        const double vl_now = net.evaluate(val, "ce");
        CHECK(near(vl_now, es->best_value()), "geri yuklenen agin val loss'u en iyi val loss'a esit olmali ("
                                              << vl_now << " vs " << es->best_value() << ")");

        // durdurulduktan sonra fit() tekrar cagrilirsa hicbir sey degismemeli
        const std::string before = net.snapshot();
        net.fit(train, 5, "ce", 0, &val);
        CHECK(net.snapshot() == before, "durdurulmus agda fit() egitime devam etmemeli");

        // val_loader olmadan early stopping hata vermeli
        NeuralNetwork n2 = make_net(21);
        n2.set_early_stopping(std::make_shared<EarlyStopping>(3));
        bool threw = false;
        try { n2.fit(train, 1, "ce", 0, nullptr); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "val_loader olmadan hata vermeli");

        bool threw2 = false;
        NeuralNetwork n3 = make_net(21);
        try { n3.restore_best_weights(); } catch (const std::logic_error&) { threw2 = true; }
        CHECK(threw2, "kayitli agirlik yokken restore_best_weights hata vermeli");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 4) Determinizm: erken duran kosu = tam kosunun ilk k epoch'u (bayt bayt) ===\n";
    {
        Problem pr = make_problem(2);

        // A: erken durdurma YOK, her epoch'ta val loss ve agirlik goruntusu kaydet
        std::vector<double> vlA; std::vector<std::string> snapA;
        {
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            for (int e = 0; e < MAXE; e++) {
                net.fit(train, 1, "ce", 0, &val);
                vlA.push_back(net.evaluate(val, "ce"));
                snapA.push_back(net.snapshot());
            }
        }
        // B: erken durdurma var, restore KAPALI -> agirliklar durma epoch'undaki gibi kalmali
        int k = 0; int best = 0;
        {
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            auto es = std::make_shared<EarlyStopping>(PAT, 0.0, /*restore_best=*/false);
            net.set_early_stopping(es);
            net.fit(train, MAXE, "ce", 0, &val);
            k = es->epochs_seen(); best = es->best_epoch();
            bool prefix = true;
            for (int i = 0; i < k; i++) prefix &= (es->history()[i] == vlA[i]);
            CHECK(prefix, "val loss gecmisi tam kosunun ilk k epoch'u ile birebir ayni olmali");
            CHECK(net.snapshot() == snapA[k - 1], "restore kapaliyken agirliklar durma epoch'undaki agirliklara esit olmali");
            // elle geri yukleme: en iyi epoch'un agirliklari, A'nin o epoch'taki agirliklari olmali
            net.restore_best_weights();
            CHECK(net.snapshot() == snapA[best], "restore_best_weights() tam olarak en iyi epoch'un agirliklarini vermeli");
        }
        // C: restore ACIK -> fit'in otomatik geri yuklemesi de ayni sonucu vermeli
        {
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            net.set_early_stopping(std::make_shared<EarlyStopping>(PAT, 0.0, true));
            net.fit(train, MAXE, "ce", 0, &val);
            CHECK(net.snapshot() == snapA[best], "otomatik geri yukleme en iyi epoch'un agirliklarini vermeli");
        }
        // D: tek epoch'luk cagrilarla (main.cpp'deki kullanim) ayni sonuc
        {
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            auto es = std::make_shared<EarlyStopping>(PAT, 0.0, false);
            net.set_early_stopping(es);
            int calls = 0;
            while (!net.should_stop() && calls < MAXE) { net.fit(train, 1, "ce", 0, &val); calls++; }
            CHECK(calls == k, "tek epoch'luk cagrilarla ayni epoch'ta durmali (" << calls << " vs " << k << ")");
            CHECK(net.snapshot() == snapA[k - 1], "tek epoch'luk cagrilarla ayni agirliklar");
        }
        std::cout << "  durma epoch'u " << k << ", en iyi epoch " << best + 1 << " (A ile karsilastirildi)\n";
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 5) Durmadan biten egitim ve scheduler ile birlikte kullanim ===\n";
    {
        Problem pr = make_problem(2);
        // patience cok buyuk: hic durmaz, son epoch agirliklari kalir; elle en iyiye donulebilir
        NeuralNetwork net = make_net(21);
        DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
        auto es = std::make_shared<EarlyStopping>(1000, 0.0, true);
        net.set_early_stopping(es);
        net.fit(train, 30, "ce", 0, &val);
        CHECK(!es->should_stop() && es->epochs_seen() == 30, "durma tetiklenmediyse 30 epoch de kosmali");
        const std::string last = net.snapshot();
        net.restore_best_weights();
        const bool moved = (net.snapshot() != last);
        CHECK(near(net.evaluate(val, "ce"), es->best_value()), "elle geri yukleme en iyi val loss'u vermeli");
        std::cout << "  30 epoch durmadan bitti; en iyi epoch " << es->best_epoch() + 1
                  << (moved ? " (son epoch'tan farkli)" : " (son epoch zaten en iyiydi)") << "\n";

        // scheduler + early stopping: durdugu epoch'ta lr = scheduler'in o epoch'a verdigi deger
        NeuralNetwork n2 = make_net(21);
        n2.set_scheduler(std::make_shared<StepLR>(0.01, 3, 0.5));
        auto es2 = std::make_shared<EarlyStopping>(PAT, 0.0, false);
        n2.set_early_stopping(es2);
        n2.fit(train, MAXE, "ce", 0, &val);
        const int last_ep = es2->epochs_seen() - 1;            // son kosulan epoch (0-tabanli)
        StepLR ref(0.01, 3, 0.5);
        CHECK(near(n2.learning_rate(), ref.lr_for_epoch(last_ep)),
              "son epoch'taki lr = scheduler'in o epoch icin degeri (" << n2.learning_rate() << " vs " << ref.lr_for_epoch(last_ep) << ")");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 6) Dogruluk modu: durdurma mantigi (BUYUK = IYI) ===\n";
    {
        EarlyStopping a(2, 0.0, true, Monitor::ValAccuracy);
        CHECK(a.maximize() && a.monitor() == Monitor::ValAccuracy && a.best_value() == -std::numeric_limits<double>::infinity(),
              "a) baslangic en iyi = -inf");
        const double m[] = {0.90, 0.92, 0.91, 0.915};
        const bool imp[] = {true, true, false, false};
        for (int i = 0; i < 4; i++) CHECK(a.on_epoch_end(m[i]) == imp[i], "a) iyilesme bayragi epoch " << i);
        CHECK(a.should_stop() && a.stopped_epoch() == 3 && a.best_epoch() == 1 && near(a.best_value(), 0.92),
              "a) epoch 3'te durmali, en iyi epoch 1 (0.92)");

        EarlyStopping b(2, 0.0, true, Monitor::ValAccuracy);          // esitlik iyilesme degildir
        for (double v : {0.9, 0.9, 0.9}) b.on_epoch_end(v);
        CHECK(b.should_stop() && b.best_epoch() == 0 && b.stopped_epoch() == 2, "b) esit dogruluk iyilesme sayilmamali");

        EarlyStopping c(2, 0.01, true, Monitor::ValAccuracy);         // min_delta: en az 0.01 artmali
        c.on_epoch_end(0.90);
        c.on_epoch_end(0.905);                                        // +0.005 < 0.01 -> iyilesme degil
        CHECK(c.best_epoch() == 0, "c) min_delta'dan kucuk artis sayilmamali");
        c.on_epoch_end(0.95);
        CHECK(c.best_epoch() == 2 && !c.should_stop(), "c) min_delta'dan buyuk artis sayilmali, bekleme sifirlanmali");

        const double nan = std::numeric_limits<double>::quiet_NaN();
        EarlyStopping d(2, 0.0, true, Monitor::ValAccuracy);
        for (double v : {0.8, nan, nan}) d.on_epoch_end(v);
        CHECK(d.should_stop() && d.best_epoch() == 0 && near(d.best_value(), 0.8), "d) NaN iyilesme sayilmamali");

        EarlyStopping e(1, 0.0, true, Monitor::ValAccuracy);
        e.on_epoch_end(0.0);                                          // %0 dogruluk bile -inf'den iyidir
        CHECK(e.best_epoch() == 0, "e) ilk (sonlu) metrik her zaman iyilesmedir");
        e.reset();
        CHECK(e.best_value() == -std::numeric_limits<double>::infinity() && e.epochs_seen() == 0, "f) reset() -inf'e donmeli");

        EarlyStopping l(2);   // varsayilan hala val loss
        CHECK(!l.maximize() && l.monitor() == Monitor::ValLoss, "g) varsayilan mod ValLoss olmali");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 7) NeuralNetwork::accuracy(): elle sayimla karsilastirma ===\n";
    {
        // cok sinifli (one-hot), N=50 batch boyutuna tam bolunmuyor
        Matrix::set_seed(31);
        NeuralNetwork net(std::make_shared<Adam>());
        net.add(std::make_unique<DenseLayer>(4, 8)); net.add(std::make_unique<TanhLayer>());
        net.add(std::make_unique<DenseLayer>(8, 3)); net.add(std::make_unique<SoftmaxLayer>());
        Matrix X = Matrix::random(4, 50), Y(3, 50, 0.0);
        std::vector<int> lab(50);
        for (int j = 0; j < 50; j++) lab[j] = (j * 7 + 3) % 3;
        Y = one_hot(lab, 3);

        net.set_training(false);
        Matrix p = net.predict(X);
        int manual = 0;
        for (int j = 0; j < 50; j++) {
            int best = 0;
            for (int i = 1; i < 3; i++) if (p.at(i, j) > p.at(best, j)) best = i;
            manual += (best == lab[j]);
        }
        DataLoader l7(X, Y, 7, false), l50(X, Y, 50, false), l1(X, Y, 1, false);
        const double a7 = net.accuracy(l7), a50 = net.accuracy(l50), a1 = net.accuracy(l1);
        std::cout << "  cok sinifli: elle " << manual << "/50 = " << std::fixed << std::setprecision(2) << manual / 50.0
                  << "  | batch 7: " << a7 << "  batch 50: " << a50 << "  batch 1: " << a1 << "\n";
        CHECK(near(a7, manual / 50.0) && near(a50, manual / 50.0) && near(a1, manual / 50.0),
              "accuracy() elle sayima ve batch boyutundan bagimsiz olmali");

        // ikili: tek sigmoid cikis, esik 0.5
        Matrix::set_seed(32);
        NeuralNetwork nb(std::make_shared<Adam>());
        nb.add(std::make_unique<DenseLayer>(4, 6)); nb.add(std::make_unique<TanhLayer>());
        nb.add(std::make_unique<DenseLayer>(6, 1)); nb.add(std::make_unique<SigmoidLayer>());
        Matrix Yb(1, 50);
        for (int j = 0; j < 50; j++) Yb.at(0, j) = (j % 3 == 0) ? 1.0 : 0.0;
        nb.set_training(false);
        Matrix pb = nb.predict(X);
        int manual_b = 0;
        for (int j = 0; j < 50; j++) manual_b += ((pb.at(0, j) > 0.5 ? 1 : 0) == (int)Yb.at(0, j));
        DataLoader lb(X, Yb, 9, false);
        CHECK(near(nb.accuracy(lb), manual_b / 50.0), "ikili dogruluk elle sayima esit olmali");
        std::cout << "  ikili: elle " << manual_b << "/50 = " << manual_b / 50.0 << "  accuracy(): " << nb.accuracy(lb) << "\n";

        // mukemmel ve en kotu durum
        DataLoader lperf(X, Y, 10, false);
        Matrix Yperf(3, 50, 0.0);
        for (int j = 0; j < 50; j++) {
            int best = 0;
            for (int i = 1; i < 3; i++) if (p.at(i, j) > p.at(best, j)) best = i;
            Yperf.at(best, j) = 1.0;
        }
        DataLoader lp(X, Yperf, 10, false);
        CHECK(near(net.accuracy(lp), 1.0), "hedef = tahmin ise dogruluk 1.0 olmali");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    // ------------------------------------------------------------
    std::cout << "\n=== 8) Dogruluk modu ag entegrasyonu + determinizm ===\n";
    {
        Problem pr = make_problem(2);

        // A: erken durdurma YOK; her epoch'ta val dogrulugu ve agirlik goruntusu
        std::vector<double> accA; std::vector<std::string> snapA;
        {
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            for (int e = 0; e < MAXE; e++) {
                net.fit(train, 1, "ce", 0, &val);
                accA.push_back(net.accuracy(val));
                snapA.push_back(net.snapshot());
            }
        }
        int k = 0, best = 0;
        {   // B: dogruluk izleniyor, restore KAPALI
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            auto es = std::make_shared<EarlyStopping>(PAT, 0.0, false, Monitor::ValAccuracy);
            net.set_early_stopping(es);
            net.fit(train, MAXE, "ce", 0, &val);
            k = es->epochs_seen(); best = es->best_epoch();
            const auto& h = es->history();
            std::cout << "  " << k << " / " << MAXE << " epoch kosuldu; en iyi epoch " << best + 1
                      << ", val dogrulugu " << std::fixed << std::setprecision(4) << es->best_value()
                      << " (durdugu epoch'ta " << h.back() << ")\n";
            CHECK(es->should_stop() && k < MAXE, "dogruluk modunda da erken durmali");
            double mx = h[0]; int mx_i = 0;
            for (size_t i = 1; i < h.size(); i++) if (h[i] > mx) { mx = h[i]; mx_i = (int)i; }
            CHECK(near(es->best_value(), mx) && best == mx_i, "best_value/best_epoch = gecmisin MAKSIMUMU olmali");
            CHECK(es->stopped_epoch() - best == PAT, "durma = en iyiden tam patience epoch sonra");
            bool prefix = true;
            for (int i = 0; i < k; i++) prefix &= (h[i] == accA[i]);
            CHECK(prefix, "dogruluk gecmisi tam kosunun ilk k epoch'u ile birebir ayni olmali");
            CHECK(net.snapshot() == snapA[k - 1], "restore kapaliyken agirliklar durma epoch'undakine esit olmali");
            net.restore_best_weights();
            CHECK(net.snapshot() == snapA[best], "restore_best_weights() en iyi (dogruluk) epoch'un agirliklarini vermeli");
            DataLoader val2(pr.Xva, pr.Yva, 100, false);
            CHECK(near(net.accuracy(val2), es->best_value()), "geri yuklenen agin val dogrulugu best_value'ya esit olmali");
        }
        {   // C: otomatik geri yukleme
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5), val(pr.Xva, pr.Yva, 100, false);
            net.set_early_stopping(std::make_shared<EarlyStopping>(PAT, 0.0, true, Monitor::ValAccuracy));
            net.fit(train, MAXE, "ce", 0, &val);
            CHECK(net.snapshot() == snapA[best], "otomatik geri yukleme en iyi epoch'un agirliklarini vermeli");
        }
        {   // D: val_loader olmadan hata
            NeuralNetwork net = make_net(21);
            DataLoader train(pr.Xtr, pr.Ytr, 20, true, false, 5);
            net.set_early_stopping(std::make_shared<EarlyStopping>(3, 0.0, true, Monitor::ValAccuracy));
            bool threw = false;
            try { net.fit(train, 1, "ce", 0, nullptr); } catch (const std::invalid_argument&) { threw = true; }
            CHECK(threw, "val_loader olmadan hata vermeli");
        }
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n" << (failures == 0 ? "TUM TESTLER GECTI" : "BASARISIZ KONTROL SAYISI: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
