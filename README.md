# C++ ile Sıfırdan Sinir Ağı

Harici makine öğrenmesi kütüphanesi kullanmadan, yalnızca standart C++17 ile yazılmış bir sinir ağı kütüphanesi: kendi matris motoru, katmanlar, optimizer'lar, eğitim döngüsü ve MNIST üzerinde uçtan uca doğrulama. Amaç, PyTorch/TensorFlow'un arka planda ne yaptığını **yazarak** anlamak.

MNIST'te küçük bir CNN ile **%99.06 ± 0.14** test doğruluğu (3 tohum, 10 epoch), bir MLP ile ~%98.2.

---

## İçindekiler
1. [Özellikler](#özellikler)
2. [Dosya yapısı](#dosya-yapısı)
3. [Derleme ve çalıştırma](#derleme-ve-çalıştırma)
4. [Testler](#testler)
5. [MNIST](#mnist)
6. [Kütüphane kullanımı](#kütüphane-kullanımı)
7. [Tasarım kuralları](#tasarım-kuralları)
8. [Ölçülen sonuçlar](#ölçülen-sonuçlar)
9. [Bilinen sınırlar](#bilinen-sınırlar)
10. [Geliştirme fikirleri](#geliştirme-fikirleri)

---

## Özellikler

| Alan | İçerik |
|---|---|
| **Matris motoru** | `Matrix` sınıfı; AVX2 / SSE2 / saf C++ çarpım çekirdekleri (çalışma anında CPU'ya göre seçilir), isteğe bağlı OpenMP |
| **Katmanlar** | `DenseLayer`, `Conv2DLayer`, `MaxPool2DLayer`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout`, `BatchNorm` |
| **Kayıp fonksiyonları** | MSE, binary cross-entropy (BCE), cross-entropy (CE) |
| **Optimizer'lar** | SGD, Adam, AdamW; L2 weight decay (yalnızca ağırlık matrislerine uygulanır) |
| **LR scheduler'lar** | Sabit, Step, Exponential, Cosine annealing, ReduceLROnPlateau |
| **Eğitim** | Mini-batch `DataLoader` (shuffle), validation izleme, **early stopping** (val loss ya da val doğruluğu, en iyi ağırlıkları geri yükler) |
| **Model yönetimi** | İkili dosyaya kaydet/yükle (BatchNorm istatistikleri dahil), bellekte `snapshot()/restore()` |
| **Veri** | MNIST IDX okuyucu, `one_hot`, sütun dilimleme |
| **Doğrulama** | Sayısal gradyan kontrolü, referans uygulamalarla karşılaştırma, mutasyon testiyle sınanmış testler |

---

## Dosya yapısı

| Dosya | Görevi |
|---|---|
| `matrix.h` / `matrix.cpp` | Matris motoru ve hızlı GEMM çekirdekleri |
| `layer.h` | `Layer` taban sınıfı, Dense, aktivasyonlar, Softmax, Dropout, BatchNorm, kayıp fonksiyonları |
| `conv.h` | `Conv2DLayer` (im2col + GEMM) ve `MaxPool2DLayer` |
| `optimizer.h` | SGD, Adam, AdamW, weight decay |
| `scheduler.h` | Learning rate scheduler'lar |
| `earlystopping.h` | `EarlyStopping` (loss veya doğruluk izleme) |
| `network.h` / `network.cpp` | `NeuralNetwork`: `predict`, `fit`, `evaluate`, `accuracy`, kaydet/yükle, snapshot |
| `dataloader.h` | Mini-batch yükleyici, `one_hot`, dilimleme |
| `mnist.h` | MNIST IDX okuyucu |
| `serialization.h` | İkili okuma/yazma yardımcıları |
| `main.cpp` | MNIST eğitim programı (`nn`) |
| `bench.cpp` | Çarpım hızı ve doğruluğu |
| `*_test.cpp` | Test paketleri (aşağıya bakın) |
| `Makefile`, `CMakeLists.txt` | Derleme |

---

## Derleme ve çalıştırma

Gereksinim: C++17 derleyici (GCC / Clang / MSVC). Harici bağımlılık yok.

### Makefile (Linux / macOS / MinGW)

```bash
make                # nn, bench ve tüm test programlarını derler
make test           # tüm testleri çalıştırır
make OMP=1          # OpenMP ile paralel çarpım
make NOSIMD=1       # SIMD çekirdeklerini kapat (sade C++)
./bench             # çarpım hızı ve referansla doğruluk karşılaştırması
```

### VS Code / CMake

1. **CMake Tools** ve **C/C++** uzantılarını kur.
2. Klasörü aç, alt çubuktan bir **kit** seç.
3. **Alt çubuktaki `Debug` yazısına tıklayıp `Release` seç.** Debug derlemesinde eğitim çok yavaş çalışır.
4. Derle (`F7`). Hedefler: `nn`, `bench`, `sched_test`, `opt_test`, `grad_test`, `es_test`, `conv_test`.

CMake seçenekleri: `NN_USE_OPENMP` (varsayılan açık), `NN_NO_SIMD`, `NN_MSVC_AVX2` (MSVC'de `/arch:AVX2`).

### Çarpım çekirdeği seçimi
Program başlarken CPU'ya bakıp AVX2 → SSE2 → saf C++ sırasıyla en iyisini seçer. Zorlamak için: `NN_MATMUL=scalar|sse2|avx2` ortam değişkeni. Üç çekirdek ve OpenMP'nin tüm thread sayıları **bit-bit aynı sonucu** üretir (derleyiciye global `-mfma` / `-march=native` verirsen derleyici çarpma+toplamayı birleştirebilir ve son basamaklar ~1e-14 değişebilir).

---

## Testler

`make test` beş paketi çalıştırır:

| Paket | Neyi doğruluyor |
|---|---|
| `grad_test` | Her katman, kayıp ve tam ağlar için analitik gradyanı **sayısal türevle** karşılaştırır (Dense, BatchNorm, Conv2D, MaxPool, Softmax+CE, tam CNN...). Kasıtlı bozuk bir katmanın yakalandığını da (negatif kontrol) sınar. |
| `conv_test` | Conv2D/MaxPool ileri geçişini im2col kullanmayan **naif bir referansla** karşılaştırır, öteleme özelliği, hata durumları, kaydet/yükle, basit bir görevde öğrenme |
| `opt_test` | SGD/Adam/AdamW ve weight decay'i elle hesaplanmış değerlerle; `wd=0` iken eski sonuçla bit-bit eşitlik |
| `sched_test` | Scheduler formülleri ve ağ entegrasyonu |
| `es_test` | Early stopping mantığı, snapshot/restore'un bit-bit doğruluğu, "erken duran koşu = tam koşunun ilk k epoch'u" determinizmi |

Bu testlerin hata yakalayabildiği **mutasyon testiyle** sınandı: koda kasıtlı hatalar sokuldu ve testlerin başarısız olduğu doğrulandı.

> **Not:** Gradyan kontrolü gerçek bir hata buldu: `DenseLayer` gradyanı fazladan `1/batch` ile bölüyordu (gradyan `1/batch²` ölçeğindeydi). Adam ölçeğe büyük ölçüde duyarsız olduğu için bu fark edilmemişti, ama Adam'daki L2 weight decay fiilen `batch` kat daha güçlüydü. Düzeltildi ve etkilenen deneyler yeniden koşuldu.

---

## MNIST

### Veriyi indir
Dört dosya gerekir. Aşağıdaki adres bir üçüncü taraf yansısıdır (orijinal kaynak: yann.lecun.com); dosya boyutları standart MNIST ile birebir aynı çıktı.

```bash
B=https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -L -o $f.gz $B/$f.gz && gunzip -f $f.gz
done
```

`.gz` dosyalarını **açman gerekir** (zlib kullanılmıyor). Python ile açmak için:

```bash
python -c "import gzip,shutil; [shutil.copyfileobj(gzip.open(f+'.gz'),open(f,'wb')) for f in ['train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']]"
```

Doğrulama: açılmış `train-images-idx3-ubyte` 47.040.016 bayt olmalı.

### Eğitim programı

```
./nn [veri_klasoru] [epoch] [batch] [max_ornek] [varyant] [scheduler] [lr0] [tohum] [optimizer] [patience] [izleme]
```

| Argüman | Değerler | Varsayılan |
|---|---|---|
| `veri_klasoru` | 4 IDX dosyasının klasörü | `.` |
| `epoch`, `batch` | tam sayı | `10`, `64` |
| `max_ornek` | eğitimde kullanılacak örnek sayısı, `0` = hepsi | `0` |
| `varyant` | `base` `drop` `bn` `bn_drop` `cnn` `cnn_drop` | `base` |
| `scheduler` | `none` `step` `exp` `cosine` `plateau` | `none` |
| `lr0` | başlangıç öğrenme oranı | `0.001` |
| `tohum` | ağırlık başlatma + karıştırma + dropout | `42` |
| `optimizer` | `adam`, `l2=<λ>`, `adamw=<wd>` | `adam` |
| `patience` | early stopping sabrı, `0` = kapalı | `0` |
| `izleme` | `loss` veya `acc` (neyi izleyeceği) | `loss` |

Veri 55.000 eğitim / 5.000 doğrulama / 10.000 test olarak bölünür. **Test seti yalnızca en sonda bir kez** kullanılır.

```bash
# hızlı deneme (~3 sn)
./nn ./mnist 1 64 3000 base

# CNN: cosine, 10 epoch
./nn ./mnist 10 64 0 cnn cosine 0.003 1 adam

# AdamW + early stopping (sabır 5)
./nn ./mnist 30 64 0 drop cosine 0.003 1 adamw=0.1 5
```

Program her epoch'u yazdırır, en iyi epoch'un modelini `mnist_<varyant>_<scheduler>_<optimizer>_s<tohum>.nn` olarak kaydeder ve betikle toplamak için tek satırlık bir `RESULT ...` özeti basar.

**Varyantlar:**
- `base`: Dense(784→128→64→10), ReLU
- `drop` / `bn` / `bn_drop`: aynısı + Dropout(0.2) / BatchNorm / ikisi birden
- `cnn`: Conv(1→8,3×3) → ReLU → Pool → Conv(8→16,3×3) → ReLU → Pool → Dense(784→64) → Dense(64→10)
- `cnn_drop`: aynısı + Dense'ten sonra Dropout(0.2)

---

## Kütüphane kullanımı

Bu örnekler derlenip çalıştırılarak doğrulandı.

**XOR, kaydet ve yükle:**
```cpp
#include "network.h"
#include "dataloader.h"

Matrix X(2, 4, {0, 0, 1, 1,
                0, 1, 0, 1});          // her SUTUN bir örnek
Matrix Y(1, 4, {0, 1, 1, 0});

NeuralNetwork net(std::make_shared<Adam>(0.01));
net.add(std::make_unique<DenseLayer>(2, 8));
net.add(std::make_unique<TanhLayer>());
net.add(std::make_unique<DenseLayer>(8, 1));
net.add(std::make_unique<SigmoidLayer>());

DataLoader train(X, Y, /*batch_size=*/4);
net.fit(train, /*epochs=*/1000, "bce", /*print_every=*/500);
net.save("xor.nn");

NeuralNetwork yeni(std::make_shared<Adam>());   // AYNI mimari kurulmalı
/* ... aynı katmanlar ... */
yeni.load("xor.nn");                            // katman adları/boyutları kontrol edilir
```

**Scheduler + early stopping + weight decay:**
```cpp
NeuralNetwork net(std::make_shared<AdamW>(0.01, /*weight_decay=*/0.05));
/* ... katmanlar ... */
net.set_scheduler(std::make_shared<CosineAnnealingLR>(0.01, /*T_max=*/100));
net.set_early_stopping(std::make_shared<EarlyStopping>(/*patience=*/10));  // val loss izler
net.fit(train, 100, "bce", /*print_every=*/0, &val);   // early stopping için val_loader ZORUNLU
double acc = net.accuracy(val);
```

**CNN (MNIST 28×28):**
```cpp
net.add(std::make_unique<Conv2DLayer>(1, 28, 28, 8, /*kernel=*/3, /*stride=*/1, /*pad=*/1)); // 8x28x28
net.add(std::make_unique<ReLULayer>());
net.add(std::make_unique<MaxPool2DLayer>(8, 28, 28, /*pool=*/2));                            // 8x14x14
net.add(std::make_unique<Conv2DLayer>(8, 14, 14, 16, 3, 1, 1));
net.add(std::make_unique<ReLULayer>());
net.add(std::make_unique<MaxPool2DLayer>(16, 14, 14, 2));                                    // 16x7x7
net.add(std::make_unique<DenseLayer>(16 * 7 * 7, 64));      // Flatten gerekmez
net.add(std::make_unique<ReLULayer>());
net.add(std::make_unique<DenseLayer>(64, 10));
net.add(std::make_unique<SoftmaxLayer>());                  // kayıp: "ce"
```

**Matris API'si (kısa):** `Matrix::random/zeros/ones`, `A * B`, `transpose()`, `hadamard()`, `apply(f)`, `sum()`, `sum_cols()`, `clip()`, `print()`, `Matrix::set_seed(s)`.

---

## Tasarım kuralları

- **Veri düzeni:** matrisler `(özellik × batch)`; her **sütun** bir örnek. Görüntüler `C×H×W` sırasıyla düzleştirilmiş sütunlar (`satır = (c·H + h)·W + w`). Conv/pool katmanları geometrilerini kurucuda alır.
- **Kayıp fonksiyonları gradyanı zaten batch'e böler.** Katmanlar bir daha bölmez (aksi halde gradyan `1/batch²` olur; yukarıdaki nota bakın).
- **Eğitim / test modu:** Dropout ve BatchNorm ikisinde farklı davranır. `fit()` bittiğinde ağ test modundadır; elle tahmin yapmadan önce `net.set_training(false)` çağır.
- **Tekrarlanabilirlik:** ağı kurmadan **önce** `Matrix::set_seed(s)` çağır; `DataLoader` ve `DropoutLayer` ayrıca `seed` alır.
- **Weight decay** yalnızca `..._W` parametrelerine uygulanır (ağırlık matrisleri); bias ve BatchNorm γ/β'ya uygulanmaz.
- **Early stopping** durunca (`restore_best` açıksa) en iyi epoch'un ağırlıklarını bellekten geri yükler. Durmadan biterse son epoch'un ağırlıkları kalır, istersen `net.restore_best_weights()` çağır.

---

## Ölçülen sonuçlar

Aşağıdaki sayılar bu projede ölçüldü: MNIST, 55k eğitim / 5k doğrulama / 10k test, batch 64, **3 tohum** (ortalama ± standart sapma). Testin kendisi ±0.14 puan gürültü taşır; küçük farkları ("0.1 puan") yorumlarken dikkatli ol.

**MLP ve CNN** (cosine, lr0=0.003, 10 epoch, son epoch):

| Ağ | Test % | Epoch süresi |
|---|---|---|
| MLP `bn_drop` | 98.17 ± 0.06 | 2.9 sn |
| `cnn` | **99.06 ± 0.14** | 19.9 sn |
| `cnn_drop` | **99.09 ± 0.14** | 19.4 sn |

CNN, MLP'yi her tohumda +0.8 ile +1.0 puan geçti. Dropout CNN'e fark etmedi (fark ≤ 0.05).

**Matris çarpımı** (bu makine: Intel Xeon 2.1 GHz, AVX2, tek çekirdek): ~3 GFLOP/s'den ~15–29 GFLOP/s'ye. MLP epoch'u 18 sn'den ~2.5 sn'ye. Ölçümler VM'de koşudan koşuya dalgalanır.

**Diğer bulgular** (hepsi tek ağ ve 3 tohumla, mütevazı etkiler):
- **Scheduler:** cosine (0.003) sabit lr'den (0.001) +0.18 puan iyi (`bn_drop`, 15 epoch, 3/3 tohumda pozitif).
- **Weight decay** (`drop`, cosine, 15 epoch): Adam 98.14, L2 (λ=1e-4) 98.24, **AdamW (wd=0.1) 98.33**. Aradaki farklar gürültü sınırında; AdamW daha tutarlı.
- **Early stopping:** hesaplamayı %37–75 azaltıyor, ama 30 epoch'u bitirip son epoch'u almaya göre ortalama 0.1–0.2 puan geride. Val **doğruluğunu** izlemek val loss'u izlemeye göre güvenilir bir üstünlük getirmedi (doğruluk gürültülü ve platolu).

---

## Bilinen sınırlar

- **Windows ve CMake doğrulanmadı.** Her şey Linux/GCC üzerinde geliştirildi ve test edildi. MinGW/MSVC derlemesi, `CMakeLists.txt` ve AVX2 çekirdeğinin Windows'taki davranışı denenmedi. Sorun olursa `NN_MATMUL=sse2` ortam değişkenini veya `-DNN_NO_SIMD` derleme bayrağını dene.
- **OpenMP hızlanması ölçülmedi** (geliştirme makinesinde tek çekirdek vardı); yalnızca sonuçların seri sürümle bit-bit aynı olduğu doğrulandı. Uzun `k`'lı `dW` çarpımı hâlâ az görev üretir.
- **`BatchNorm2D` yok.** Mevcut `BatchNorm` her konumu ayrı özellik sayar (kanal başına normalleştirme yapmaz); CNN deneylerinde kullanılmadı.
- **Mimari model dosyasında saklanmaz.** `load()` için aynı mimariyi elle kurmak gerekir (katman adı ve boyutları kontrol edilir). Adam'ın momentum durumu kaydedilmez, yani eğitim kaldığı yerden birebir devam etmez.
- Model dosyaları yazıldığı makinenin bayt sırasını kullanır; farklı mimariler arası taşınabilirlik hedeflenmedi.
- Eski `vector<Matrix>` tabanlı `fit()` scheduler ve early stopping desteklemez; `DataLoader` tabanlı sürümü kullan.
- Sadece `double` (float32 yok), CPU, tek veri seti (MNIST), küçük mimariler.
- CNN epoch'u ~20 sn: GEMM çekirdeği verimli, kalan zaman im2col, bellek ayırma ve elementwise işlerde.

---

## Geliştirme fikirleri

- `BatchNorm2D`, veri artırma (kaydırma/döndürme), `float` desteği
- Mimariyi dosyada saklayan tam model kaydı ve Adam durumunu kaydetme
- CSV veri yükleyici (tablolu veri)
- SGD momentum / RMSProp
- im2col ve bellek ayırma maliyetini azaltma, `dW` için k-boyutunda paralellik
- Windows ve çok çekirdekli makinede doğrulama
