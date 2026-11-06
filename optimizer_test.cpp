#include "network.h"
#include "optimizer.h"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "  [HATA] " << msg << "\n"; failures++; } } while (0)
static bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }
static bool same_bits(const Matrix& a, const Matrix& b) {
    return a.data.size() == b.data.size() &&
           std::memcmp(a.data.data(), b.data.data(), a.data.size() * sizeof(double)) == 0;
}

int main() {
    std::cout << "=== 1) SGD + L2: elle hesaplanmis degerler ===\n";
    {
        // theta -= lr * (g + wd*theta);  lr=0.1, wd=0.5
        SGD sgd(0.1, 0.5);
        Matrix p(1, 2, {1.0, -2.0}), g(1, 2, {0.0, 0.0});
        sgd.step(p, g, "layer0_W");
        CHECK(near(p.data[0], 0.95) && near(p.data[1], -1.9), "grad=0: W kuculmeli -> [0.95, -1.9]");

        Matrix q(1, 2, {1.0, -2.0}), g2(1, 2, {1.0, 1.0});
        sgd.step(q, g2, "layer0_W");
        CHECK(near(q.data[0], 0.85) && near(q.data[1], -2.0), "grad=1: [1 - 0.1*(1+0.5), -2 - 0.1*(1-1)]");

        Matrix b(1, 2, {1.0, -2.0}), gb(1, 2, {0.0, 0.0});
        sgd.step(b, gb, "layer0_b");
        CHECK(near(b.data[0], 1.0) && near(b.data[1], -2.0), "bias'a decay uygulanmamali");

        SGD plain(0.1, 0.0);
        Matrix r(1, 2, {1.0, -2.0}), gr(1, 2, {1.0, 1.0});
        plain.step(r, gr, "layer0_W");
        CHECK(near(r.data[0], 0.9) && near(r.data[1], -2.1), "wd=0: duz SGD");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 2) AdamW ve Adam+L2 fark ediyor mu? (gradyan = 0, alpha=0.1, wd=0.5) ===\n";
    {
        Matrix g(1, 2, {0.0, 0.0});

        AdamW adamw(0.1, 0.5);
        Matrix pw(1, 2, {1.0, -2.0});
        adamw.step(pw, g, "layer0_W");
        // AdamW: theta *= (1 - alpha*wd) = 0.95 ; gradyan 0 oldugu icin Adam adimi 0
        CHECK(near(pw.data[0], 0.95) && near(pw.data[1], -1.9), "AdamW: theta*(1-alpha*wd) = [0.95, -1.9]");

        Adam l2(0.1, 0.9, 0.999, 1e-8, 0.5, /*decoupled=*/false);
        Matrix pl(1, 2, {1.0, -2.0});
        l2.step(pl, g, "layer0_W");
        // L2: g_eff = 0.5*theta -> Adam adimi ~ alpha*sign(theta): [0.9, -1.9]
        // (buyuk |theta| icin bile adim ayni buyuklukte: Adam gradyani normalize eder)
        CHECK(near(pl.data[0], 0.9, 1e-6) && near(pl.data[1], -2.0 + 0.1, 1e-6),
              "Adam+L2: ~ theta - alpha*sign(theta) = [0.9, -1.9]");
        std::cout << "  AdamW      : [" << pw.data[0] << ", " << pw.data[1] << "]  (agirlikla orantili kuculur)\n";
        std::cout << "  Adam + L2  : [" << std::fixed << std::setprecision(4) << pl.data[0] << ", " << pl.data[1]
                  << "]  (agirliktan bagimsiz sabit adim: Adam normalize eder)\n";

        for (const char* id : {"layer1_b", "layer1_gamma", "layer1_beta"}) {
            AdamW a(0.1, 0.5);
            Matrix p(1, 2, {1.0, -2.0});
            a.step(p, g, id);
            CHECK(near(p.data[0], 1.0) && near(p.data[1], -2.0), std::string("decay uygulanmamali: ") + id);
        }
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 3) wd = 0 iken eski Adam ile bit-bit ayni mi? ===\n";
    {
        Matrix::set_seed(5);
        Adam   a0 (0.003);                                     // eski kullanim
        Adam   a1 (0.003, 0.9, 0.999, 1e-8, 0.0, false);       // L2 kapali
        AdamW  a2 (0.003, 0.0);                                // AdamW, wd=0
        Matrix p0 = Matrix::random(6, 5), p1 = p0, p2 = p0;
        for (int t = 0; t < 25; t++) {
            Matrix g = Matrix::random(6, 5);
            a0.step(p0, g, "layer0_W");
            a1.step(p1, g, "layer0_W");
            a2.step(p2, g, "layer0_W");
        }
        CHECK(same_bits(p0, p1), "Adam(L2, wd=0) != Adam");
        CHECK(same_bits(p0, p2), "AdamW(wd=0) != Adam");

        SGD s0(0.05), s1(0.05, 0.0);
        Matrix q0 = Matrix::random(4, 4), q1 = q0;
        for (int t = 0; t < 10; t++) {
            Matrix g = Matrix::random(4, 4);
            s0.step(q0, g, "layer0_W"); s1.step(q1, g, "layer0_W");
        }
        CHECK(same_bits(q0, q1), "SGD(wd=0) != SGD");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 4) Scheduler ile birlikte: AdamW decay'i guncel lr ile olcekler ===\n";
    {
        AdamW a(0.1, 0.5);
        Matrix g(1, 1, {0.0}), p1(1, 1, {1.0}), p2(1, 1, {1.0});
        a.step(p1, g, "layer0_W");              // lr=0.1  -> 1*(1-0.05) = 0.95
        a.set_learning_rate(0.01);
        a.step(p2, g, "layer0_W");              // lr=0.01 -> 1*(1-0.005) = 0.995
        CHECK(near(p1.data[0], 0.95) && near(p2.data[0], 0.995), "decay guncel lr ile olceklenmeli");
        std::cout << "  lr=0.1 -> " << std::setprecision(3) << p1.data[0] << ",  lr=0.01 -> " << p2.data[0] << "\n";
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n=== 5) Ag seviyesinde: weight decay agirlik normunu kuculturmeli ===\n";
    {
        Matrix X(2, 4, {0,0,1,1, 0,1,0,1});
        Matrix Y(1, 4, {0,1,1,0});
        auto run = [&](std::shared_ptr<Optimizer> opt) {
            Matrix::set_seed(7);
            NeuralNetwork net(opt);
            net.add(std::make_unique<DenseLayer>(2, 16));
            net.add(std::make_unique<TanhLayer>());
            net.add(std::make_unique<DenseLayer>(16, 1));
            net.add(std::make_unique<SigmoidLayer>());
            std::vector<Matrix> xb = {X}, yb = {Y};
            net.fit(xb, yb, 600, "bce", 100000);
            return net.weight_norm();
        };
        double n0 = run(std::make_shared<Adam>(0.01));
        double n1 = run(std::make_shared<AdamW>(0.01, 1.0));
        std::cout << "  ||W||  Adam: " << std::setprecision(4) << n0 << "   AdamW(wd=1.0): " << n1 << "\n";
        CHECK(n1 < n0, "AdamW agirlik normunu kucultmeli");
    }
    std::cout << (failures ? "  basarisiz\n" : "  tamam\n");

    std::cout << "\n" << (failures == 0 ? "TUM TESTLER GECTI" : "BASARISIZ TEST SAYISI: " + std::to_string(failures)) << "\n";
    return failures == 0 ? 0 : 1;
}
