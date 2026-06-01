#include "network.h"
#include <iostream>
#include <iomanip>

// XOR dataseti: (2 x 4)
Matrix make_X() { return Matrix(2, 4, {0,0,1,1, 0,1,0,1}); }
Matrix make_Y() { return Matrix(1, 4, {0,1,1,0}); }

void print_predictions(NeuralNetwork& net, const Matrix& X, const Matrix& Y) {
    net.set_training(false);   // test modu — dropout kapalı
    Matrix pred = net.predict(X);
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        double p = pred.at(0, i);
        int expected = (int)Y.at(0, i);
        int got      = p > 0.5 ? 1 : 0;
        if (got == expected) correct++;
        std::cout << "    XOR(" << (int)X.at(0,i) << "," << (int)X.at(1,i) << ")"
                  << " → " << std::fixed << std::setprecision(4) << p
                  << "  (" << got << ")"
                  << (got == expected ? " ✓" : " ✗") << "\n";
    }
    std::cout << "  Doğruluk: " << correct << "/4\n";
}

int main() {
    Matrix X = make_X();
    Matrix Y = make_Y();
    std::vector<Matrix> Xb = {X}, Yb = {Y};

    // ─────────────────────────────────────────────
    // 1) Vanilla ağ (baseline)
    // ─────────────────────────────────────────────
    {
        std::cout << "\n========================================\n";
        std::cout << "  1) Vanilla (Dense → Tanh → Dense → Sigmoid)\n";
        std::cout << "========================================\n";

        NeuralNetwork net(std::make_shared<Adam>());
        net.add(std::make_unique<DenseLayer>(2, 16));
        net.add(std::make_unique<TanhLayer>());
        net.add(std::make_unique<DenseLayer>(16, 1));
        net.add(std::make_unique<SigmoidLayer>());
        net.print_summary();

        net.fit(Xb, Yb, 1000, "bce", 250);
        print_predictions(net, X, Y);
    }

    // ─────────────────────────────────────────────
    // 2) Dropout eklenmiş ağ
    //    Dense → ReLU → Dropout(0.3) → Dense → Sigmoid
    //    Not: XOR çok küçük bir problem, dropout burada
    //    kasıtlı olarak zorlaştırır — davranışı görmek için
    // ─────────────────────────────────────────────
    {
        std::cout << "\n========================================\n";
        std::cout << "  2) Dropout(p=0.3) eklenmiş\n";
        std::cout << "========================================\n";

        NeuralNetwork net(std::make_shared<Adam>());
        net.add(std::make_unique<DenseLayer>(2, 16));
        net.add(std::make_unique<ReLULayer>());
        net.add(std::make_unique<DropoutLayer>(0.3));   // %30 nöronu sıfırla
        net.add(std::make_unique<DenseLayer>(16, 1));
        net.add(std::make_unique<SigmoidLayer>());
        net.print_summary();

        net.fit(Xb, Yb, 2000, "bce", 500);
        print_predictions(net, X, Y);
    }

    // ─────────────────────────────────────────────
    // 3) BatchNorm + Dropout birlikte
    //    Dense → BatchNorm → ReLU → Dropout(0.3) → Dense → Sigmoid
    //    Tipik modern ağ bloğu sırası:
    //    Linear → BatchNorm → Aktivasyon → Dropout
    // ─────────────────────────────────────────────
    {
        std::cout << "\n========================================\n";
        std::cout << "  3) BatchNorm + Dropout birlikte\n";
        std::cout << "========================================\n";

        NeuralNetwork net(std::make_shared<Adam>());
        net.add(std::make_unique<DenseLayer>(2, 16));
        net.add(std::make_unique<BatchNormLayer>(16));  // 16 feature normalize et
        net.add(std::make_unique<ReLULayer>());
        net.add(std::make_unique<DropoutLayer>(0.3));
        net.add(std::make_unique<DenseLayer>(16, 1));
        net.add(std::make_unique<SigmoidLayer>());
        net.print_summary();

        net.fit(Xb, Yb, 2000, "bce", 500);
        print_predictions(net, X, Y);
    }

    // ─────────────────────────────────────────────
    // Dropout davranışını göster:
    // eğitim ve test modu arasındaki fark
    // ─────────────────────────────────────────────
    {
        std::cout << "\n========================================\n";
        std::cout << "  Dropout eğitim vs test modu farkı\n";
        std::cout << "========================================\n";

        DropoutLayer drop(0.5);
        Matrix test_input(4, 1, {1.0, 1.0, 1.0, 1.0});

        std::cout << "  Girdi: [1, 1, 1, 1]\n";

        drop.set_training(true);
        Matrix train_out = drop.forward(test_input);
        std::cout << "  Eğitim modu (scale=2.0, %50 sıfırlanır):\n  ";
        for (int i = 0; i < 4; i++)
            std::cout << std::setprecision(1) << std::fixed
                      << train_out.at(i,0) << " ";
        std::cout << "\n";

        drop.set_training(false);
        Matrix eval_out = drop.forward(test_input);
        std::cout << "  Test modu (pass-through):\n  ";
        for (int i = 0; i < 4; i++)
            std::cout << eval_out.at(i,0) << " ";
        std::cout << "\n";
    }

    return 0;
}
