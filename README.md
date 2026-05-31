# C++ Neural Network — Sıfırdan

Herhangi bir makine öğrenmesi kütüphanesi kullanmadan, yalnızca standart C++17 ile yazılmış tam bir sinir ağı ve matris motoru. PyTorch veya TensorFlow'un arka planda ne yaptığını anlamak amacıyla geliştirilmiştir.

---

## Özellikler

- Sıfırdan yazılmış `Matrix` sınıfı — matris çarpımı, transpoz, element-wise operasyonlar
- Abstract `Layer` sistemi — kolayca genişletilebilir katman mimarisi
- `DenseLayer` (fully connected), `ReLU`, `Sigmoid`, `Tanh` aktivasyonları
- `MSELoss` ve `BCELoss` kayıp fonksiyonları
- `Adam` optimizer (Kingma & Ba, 2014) — bias correction dahil, tam implementasyon
- `SGD` optimizer — karşılaştırma amaçlı
- XOR problemi ile doğrulanmış çalışan eğitim döngüsü

---

## Dosya Yapısı

```
.
├── matrix.h / matrix.cpp     # Matris motoru
├── layer.h                   # Layer base class + DenseLayer + aktivasyonlar + loss
├── optimizer.h               # Adam ve SGD
├── network.h / network.cpp   # NeuralNetwork sınıfı
├── main.cpp                  # XOR demo — Adam vs SGD karşılaştırması
├── CMakeLists.txt            # VS Code / CMake için
└── Makefile                  # Linux/macOS terminal için
```

---

## Derleme ve Çalıştırma

### VS Code (Windows/Linux/macOS)

1. **CMake Tools** ve **C/C++** uzantılarını yükle (`Ctrl+Shift+X`)
2. `File → Open Folder` ile proje klasörünü aç
3. Alttaki mavi bar'da **No Kit Selected** yazısına tıkla → GCC veya MSVC seç
4. `F7` ile derle, `Shift+F5` ile çalıştır

### Terminal (Linux / macOS)

```bash
make && ./nn
```

### Terminal (Manuel)

```bash
g++ -std=c++17 -O2 -o nn main.cpp matrix.cpp network.cpp
./nn          # Linux/macOS
nn.exe        # Windows
```

---

## Mimari

```
Matrix Engine
    └── Layer (abstract)
            ├── DenseLayer    →  forward: W*x + b  |  backward: dW, db, dx
            ├── ReLULayer     →  forward: max(0,x) |  backward: mask
            ├── SigmoidLayer  →  forward: 1/(1+e⁻ˣ)
            └── TanhLayer     →  forward: tanh(x)

Optimizer (abstract)
    ├── Adam
    └── SGD

NeuralNetwork
    ├── add(Layer)
    ├── predict(x)         →  forward pass
    ├── train_step(x, y)   →  forward → loss → backward → update
    └── fit(X, Y, epochs)  →  tam eğitim döngüsü
```

---

## Kullanım

```cpp
#include "network.h"

// Adam optimizer ile ağ oluştur (varsayılan: α=0.001, β1=0.9, β2=0.999, ε=1e-8)
NeuralNetwork net(std::make_shared<Adam>());

// Katman ekle
net.add(std::make_unique<DenseLayer>(2, 8));
net.add(std::make_unique<TanhLayer>());
net.add(std::make_unique<DenseLayer>(8, 1));
net.add(std::make_unique<SigmoidLayer>());

// Eğit
net.fit(X_batches, Y_batches, /*epochs=*/1000, /*loss=*/"bce", /*print_every=*/100);

// Tahmin
Matrix result = net.predict(input);
```

Optimizer'ı özelleştirmek için:

```cpp
// Özel Adam
auto opt = std::make_shared<Adam>(/*alpha=*/0.01, /*beta1=*/0.9,
                                   /*beta2=*/0.999, /*eps=*/1e-8);

// SGD
auto opt = std::make_shared<SGD>(/*lr=*/0.1);

NeuralNetwork net(opt);
```

---

## Matrix API

```cpp
Matrix A = Matrix::random(3, 3, 0.0, 1.0);  // Normal dağılım
Matrix B = Matrix::zeros(3, 3);
Matrix C = Matrix::ones(3, 3);

A * B            // Matris çarpımı
A.transpose()    // Transpoz
A.hadamard(B)    // Element-wise çarpım
A.apply(f)       // Her elemana f uygula
A.sum()          // Tüm elemanların toplamı
A.sum_cols()     // Sütunlar boyunca topla → (rows x 1)
A.clip(-1, 1)    // Değerleri sınırla
A.print("A")     // Konsola yazdır
```

---

## Adam Optimizer

Kingma & Ba (2014) makalesindeki formülün birebir C++ implementasyonu.

```
m_t = β1 * m_{t-1} + (1 - β1) * g          # 1. moment (ortalama)
v_t = β2 * v_{t-1} + (1 - β2) * g²         # 2. moment (varyans)
m̂   = m_t / (1 - β1^t)                     # bias düzeltmesi
v̂   = v_t / (1 - β2^t)                     # bias düzeltmesi
θ   = θ - α * m̂ / (√v̂ + ε)
```

| Parametre | Değer    | Açıklama                        |
|-----------|----------|---------------------------------|
| α         | 0.001    | Öğrenme oranı                   |
| β1        | 0.9      | 1. moment bozunma katsayısı     |
| β2        | 0.999    | 2. moment bozunma katsayısı     |
| ε         | 1e-8     | Sayısal kararlılık sabiti       |

Her ağırlık matrisi (`W` ve `b`) için ayrı moment geçmişi tutulur. Bias correction özellikle eğitimin ilk adımlarında kritiktir — olmadan moment tahminleri sıfıra yakın kalır ve güncellemeler çok küçük olur.

---

## Demo Çıktısı (XOR)

```
=== Network Summary ===
  Layer 0: Dense(2->8)
  Layer 1: Tanh
  Layer 2: Dense(8->1)
  Layer 3: Sigmoid
  Optimizer: Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

Epoch     1 | Loss: 0.693142
Epoch   100 | Loss: 0.521837
...
Epoch  1000 | Loss: 0.201823

  Tahminler:
    XOR(0,0) → 0.0312 (0) ✓
    XOR(0,1) → 0.9701 (1) ✓
    XOR(1,0) → 0.9698 (1) ✓
    XOR(1,1) → 0.0289 (0) ✓
  Doğruluk: 4/4
```

---

## Gereksinimler

- C++17 veya üzeri
- CMake 3.15+ (VS Code için) ya da Make (terminal için)
- Harici bağımlılık yok

---

## Geliştirme Fikirleri

- Mini-batch training ve data loader
- Adam'dan daha gelişmiş optimizerlar (AdaGrad, RMSProp)
- Softmax + cross-entropy (çok sınıflı problemler)
- Model ağırlıklarını dosyaya kaydetme/yükleme
- Conv2D katmanı
