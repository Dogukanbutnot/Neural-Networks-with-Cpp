#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// ============================================================
//  EarlyStopping
//
//  Izlenen metrik (genelde validation loss) `patience` epoch ust uste
//  iyilesmezse egitimi durdurur. Egitim loss'u dusmeye devam etse bile
//  validation loss yukseliyorsa ag ezberlemeye baslamistir.
//
//  Izlenen metrik (Monitor):
//    ValLoss     : validation loss      -> KUCUK = IYI   (varsayilan)
//    ValAccuracy : validation dogrulugu -> BUYUK = IYI   (0..1 arasi oran)
//
//  Kurallar (epoch'lar 0'dan baslar):
//    - iyilesme: loss icin  metrik < best - min_delta ; dogruluk icin metrik > best + min_delta
//                -> best guncellenir, bekleme sifirlanir
//    - aksi halde (esitlik ve NaN dahil) -> bekleme++ ;  bekleme >= patience ise DUR
//
//  DIKKAT (dogruluk modu): dogruluk ayrik bir degerdir (ornegin 5000 ornekte
//  0.0002 adimlarla) ve platoda sik sik ESIT kalir. Esitlik iyilesme sayilmaz,
//  yani esit epoch'lar sabri tuketir. min_delta'yi 0.0 birakmak en katisidir.
//
//  restore_best = true ise NeuralNetwork::fit(), durdugunda en iyi epoch'un
//  agirliklarini otomatik geri yukler. (Egitim durdurulmadan biterse son
//  epoch'un agirliklari kalir; istersen net.restore_best_weights() cagir.)
//
//  Not: LR scheduler ile birlikte kullanirken dikkat: kosinus gibi onceden
//  planlanmis bir egri, egitim erken bitince sonuna ulasmaz. Plateau veya
//  sabit lr ile daha dogal calisir.
// ============================================================
enum class Monitor { ValLoss, ValAccuracy };

class EarlyStopping {
    int     pat;
    double  min_delta_;
    bool    auto_restore;
    Monitor mon;

    double best_    = std::numeric_limits<double>::infinity();
    int    best_ep  = -1;
    int    wait_    = 0;
    int    stop_ep  = -1;
    int    epoch_   = 0;
    bool   stop_    = false;
    std::vector<double> hist;

    double worst_value() const {
        return maximize() ? -std::numeric_limits<double>::infinity()
                          :  std::numeric_limits<double>::infinity();
    }

public:
    explicit EarlyStopping(int patience, double min_delta = 0.0, bool restore_best = true,
                           Monitor monitor = Monitor::ValLoss)
        : pat(patience), min_delta_(min_delta), auto_restore(restore_best), mon(monitor) {
        if (patience < 1)     throw std::invalid_argument("EarlyStopping: patience >= 1 olmali");
        if (min_delta < 0.0)  throw std::invalid_argument("EarlyStopping: min_delta >= 0 olmali");
        best_ = worst_value();
    }

    // Epoch sonu metrigi ver. Bu epoch yeni en iyi ise true doner.
    bool on_epoch_end(double metric) {
        hist.push_back(metric);
        const int e = epoch_++;
        const bool improved = maximize() ? metric > best_ + min_delta_    // NaN -> false
                                         : metric < best_ - min_delta_;
        if (improved) {
            best_ = metric; best_ep = e; wait_ = 0;
        } else if (++wait_ >= pat) {
            stop_ = true;
            if (stop_ep < 0) stop_ep = e;
        }
        return improved;
    }

    Monitor monitor()      const { return mon; }
    bool   maximize()      const { return mon == Monitor::ValAccuracy; }
    bool   should_stop()   const { return stop_; }
    bool   restore_best()  const { return auto_restore; }
    double best_value()    const { return best_; }
    int    best_epoch()    const { return best_ep; }       // 0-tabanli, yoksa -1
    int    stopped_epoch() const { return stop_ep; }       // 0-tabanli, durmadiysa -1
    int    epochs_seen()   const { return epoch_; }
    int    patience()      const { return pat; }
    const std::vector<double>& history() const { return hist; }

    void reset() {
        best_ = worst_value();
        best_ep = stop_ep = -1; wait_ = epoch_ = 0; stop_ = false; hist.clear();
    }
};
