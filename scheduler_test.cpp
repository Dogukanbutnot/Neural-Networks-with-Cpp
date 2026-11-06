#include "network.h"
#include "scheduler.h"
#include "dataloader.h"
#include <iostream>
#include <iomanip>
#include <cmath>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "  [HATA] " << msg << "\n"; failures++; } } while (0)
static bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

int main() {
    std::cout << "=== 1) Formul kontrolleri (elle hesaplanmis degerlerle) ===\n";
    {
        StepLR s(0.1, 3, 0.1);
        CHECK(near(s.lr_for_epoch(0), 0.1)   && near(s.lr_for_epoch(2), 0.1),   "StepLR ilk blok");
        CHECK(near(s.lr_for_epoch(3), 0.01)  && near(s.lr_for_epoch(5), 0.01),  "StepLR ikinci blok");
        CHECK(near(s.lr_for_epoch(6), 0.001), "StepLR ucuncu blok");

        ExponentialLR e(1.0, 0.5);
        CHECK(near(e.lr_for_epoch(0), 1.0) && near(e.lr_for_epoch(1), 0.5) && near(e.lr_for_epoch(3), 0.125),
              "ExponentialLR");

        CosineAnnealingLR c(1.0, 10, 0.0);
        CHECK(near(c.lr_for_epoch(0), 1.0),  "Cosine baslangic = base");
        CHECK(near(c.lr_for_epoch(5), 0.5),  "Cosine orta nokta = (base+min)/2");
        CHECK(near(c.lr_for_epoch(10), 0.0, 1e-12), "Cosine T_max'ta = eta_min");
        CHECK(near(c.lr_for_epoch(25), 0.0, 1e-12), "Cosine T_max sonrasi sabit kalir");
        bool mono = true;
        for (int i = 1; i <= 10; i++) mono &= c.lr_for_epoch(i) <= c.lr_for_epoch(i - 1);
        CHECK(mono, "Cosine monoton azalir");
        CosineAnnealingLR c2(0.1, 4, 0.01);
        CHECK(near(c2.lr_for_epoch(4), 0.01), "Cosine eta_min'e iner");
        CHECK(near(c2.lr_for_epoch(2), 0.055), "Cosine eta_min varken orta nokta");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 2) ReduceLROnPlateau davranisi ===\n";
    {
        // factor 0.5, patience 1: iyilesmeyen 2. ust uste epoch'ta lr yarilanir
        ReduceLROnPlateau p(1.0, 0.5, 1, 1e-3);
        const double metrics[] = {1.0, 0.9, 0.9, 0.9, 0.9, 0.8};
        const double expect[]  = {1.0, 1.0, 1.0, 0.5, 0.5, 0.5};   // her on_epoch_end SONRASI lr
        for (int i = 0; i < 6; i++) {
            p.on_epoch_end(metrics[i]);
            CHECK(near(p.lr_for_epoch(i + 1), expect[i]),
                  "Plateau adim " << i << ": beklenen " << expect[i] << ", gelen " << p.lr_for_epoch(i + 1));
        }
        // min_lr tabani
        ReduceLROnPlateau q(1.0, 0.1, 0, 0.05);
        for (int i = 0; i < 10; i++) q.on_epoch_end(5.0);   // hic iyilesmiyor
        CHECK(near(q.lr_for_epoch(0), 0.05), "Plateau min_lr'nin altina inmemeli");
        CHECK(q.needs_metric(), "Plateau metrik ister");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 3) Gecersiz parametreler hata vermeli ===\n";
    {
        auto throws = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        CHECK(throws([]{ StepLR s(0.1, 0); }),                   "StepLR step_size=0");
        CHECK(throws([]{ CosineAnnealingLR c(0.1, 0); }),        "Cosine T_max=0");
        CHECK(throws([]{ ReduceLROnPlateau p(0.1, 1.5); }),      "Plateau factor>=1");
        CHECK(throws([]{ make_scheduler("bilinmeyen", 0.1, 10); }), "bilinmeyen isim");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 4) Ag entegrasyonu: lr gercekten optimizer'a ulasiyor mu? ===\n";
    {
        Matrix X(2, 4, {0,0,1,1, 0,1,0,1});
        Matrix Y(1, 4, {0,1,1,0});
        DataLoader loader(X, Y, 4, false);

        NeuralNetwork net(std::make_shared<Adam>(0.5));   // baslangic degeri onemsiz, scheduler ezer
        net.add(std::make_unique<DenseLayer>(2, 4));
        net.add(std::make_unique<TanhLayer>());
        net.add(std::make_unique<DenseLayer>(4, 1));
        net.add(std::make_unique<SigmoidLayer>());
        net.set_scheduler(std::make_shared<StepLR>(0.01, 2, 0.5));

        // fit'i tek epoch'luk cagrilarla calistir: sayac cagrilar arasinda korunmali
        const double expect[] = {0.01, 0.01, 0.005, 0.005, 0.0025};
        std::cout << "  epoch: lr (beklenen)\n";
        for (int e = 0; e < 5; e++) {
            net.fit(loader, 1, "bce", 0);
            std::cout << "    " << e << ": " << std::setprecision(6) << net.learning_rate()
                      << " (" << expect[e] << ")\n";
            CHECK(near(net.learning_rate(), expect[e]), "ag lr epoch " << e);
        }
        net.set_scheduler(nullptr);                // scheduler kaldirilinca lr oldugu gibi kalir
        double keep = net.learning_rate();
        net.fit(loader, 1, "bce", 0);
        CHECK(near(net.learning_rate(), keep), "scheduler yokken lr degismemeli");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 5) Karsilastirmada kullanilan ayarlar: 15 epoch, lr0 = 0.003 ===\n";
    {
        const char* kinds[] = {"none", "step", "exp", "cosine"};
        std::cout << "  epoch ";
        for (auto k : kinds) std::cout << std::setw(10) << k;
        std::cout << "\n";
        for (int e = 0; e < 15; e++) {
            std::cout << "  " << std::setw(5) << e + 1 << " ";
            for (auto k : kinds) {
                auto s = make_scheduler(k, 0.003, 15);
                std::cout << std::setw(10) << std::fixed << std::setprecision(5) << s->lr_for_epoch(e);
            }
            std::cout << "\n";
        }
        std::cout << "  (plateau metrige bagli oldugu icin tabloda yok)\n";
    }

    std::cout << "\n" << (failures == 0 ? "TUM TESTLER GECTI" : "BASARISIZ TEST SAYISI: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
