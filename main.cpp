#include "network.h"
#include <iostream>
#include <iomanip>

// XOR dataseti: (2 x 4), her sütun bir örnek
Matrix make_X() { return Matrix(2, 4, {0,0,1,1, 0,1,0,1}); }
Matrix make_Y() { return Matrix(1, 4, {0,1,1,0}); }

void run_xor(const std::string& label,
             std::shared_ptr<Optimizer> opt,
             int epochs) {
    Matrix X = make_X();
    Matrix Y = make_Y();

    NeuralNetwork net(opt);
    net.add(std::make_unique<DenseLayer>(2, 8));
    net.add(std::make_unique<TanhLayer>());
    net.add(std::make_unique<DenseLayer>(8, 1));
    net.add(std::make_unique<SigmoidLayer>());

    std::cout << "\n========================================\n";
    std::cout << "  " << label << "\n";
    std::cout << "========================================\n";
    net.print_summary();

    std::vector<Matrix> Xb = {X}, Yb = {Y};
    net.fit(Xb, Yb, epochs, "bce", epochs / 10);

    // Tahminler
    std::cout << "\n  Tahminler:\n";
    Matrix pred = net.predict(X);
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        double p = pred.at(0, i);
        int expected = (i == 1 || i == 2) ? 1 : 0;
        int predicted = p > 0.5 ? 1 : 0;
        if (predicted == expected) correct++;
        std::cout << "    XOR(" << (int)X.at(0,i) << "," << (int)X.at(1,i) << ")"
                  << " → " << std::fixed << std::setprecision(4) << p
                  << " (" << predicted << ")"
                  << (predicted == expected ? " ✓" : " ✗") << "\n";
    }
    std::cout << "  Doğruluk: " << correct << "/4\n";
}

int main() {
    // ---- Adam (varsayılan hiperparametreler) ----
    run_xor("Adam  α=0.001  β1=0.9  β2=0.999  ε=1e-8",
            std::make_shared<Adam>(0.001, 0.9, 0.999, 1e-8),
            1000);

    // ---- SGD (karşılaştırma) ----
    run_xor("SGD   lr=0.1",
            std::make_shared<SGD>(0.1),
            1000);

    // ---- Adam hızlı (daha büyük lr) ----
    run_xor("Adam  α=0.01  (daha agresif)",
            std::make_shared<Adam>(0.01, 0.9, 0.999, 1e-8),
            1000);

    return 0;
}
