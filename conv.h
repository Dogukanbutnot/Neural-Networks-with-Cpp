#pragma once
#include "layer.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
//  Conv2D ve MaxPool2D katmanlari
//
//  VERI DUZENI: agdaki her SUTUN bir ornektir. Bir goruntu, (kanal, yukseklik,
//  genislik) siralamasiyla (CHW) duzlestirilmis tek bir sutun olarak tasinir:
//      satir = (c * H + h) * W + w
//  Bu yuzden ayri bir "Flatten" katmani gerekmez: Dense dogrudan baglanir.
//
//  Conv2D = im2col + GEMM (mevcut hizli carpim cekirdegi yeniden kullanilir)
//
//    col : (K x P*B)   K = in_c*kh*kw ,  P = out_h*out_w ,  sutun indeksi = p*B + b
//    Y   = W (out_c x K) * col  ->  (out_c x P*B)
//
//  "p*B + b" sutun siralamasi iki seyi bedavaya getirir:
//   * im2col'un her satir parcasi girdi matrisinin BIR SATIRININ ardisik
//     kopyasidir (batch en hizli degisen indekstir) -> memcpy.
//   * Y'nin bellek duzeni, istedigimiz (out_c*P x B) cikti matrisiyle BIREBIR
//     aynidir; yeniden duzenleme (permutasyon) gerekmez.
//
//  Geri geciste:
//    db  = Y gradyaninin satir toplamlari
//    dW  = ( col * dY^T )^T        (K x out_c) -> (out_c x K)
//    dX  = col2im( W^T * dY )      (ilk katmanda atlanir)
// ============================================================

class Conv2DLayer : public Layer {
    int in_c, in_h, in_w, out_c, kh, kw, stride, pad;
    int out_h, out_w, K, P;
    Matrix Wm, b;          // agirliklar (out_c x K), bias (out_c x 1)
    Matrix dW, db;
    Matrix col;            // im2col onbellegi (geri gecis icin)
    int    batch = 0;
    bool   need_dx = true;

    void validate() const {
        if (in_c <= 0 || in_h <= 0 || in_w <= 0 || out_c <= 0)
            throw std::invalid_argument("Conv2D: kanal/boyutlar > 0 olmali");
        if (kh <= 0 || kw <= 0) throw std::invalid_argument("Conv2D: cekirdek boyutu > 0 olmali");
        if (stride <= 0)        throw std::invalid_argument("Conv2D: stride > 0 olmali");
        if (pad < 0)            throw std::invalid_argument("Conv2D: pad >= 0 olmali");
        if (kh > in_h + 2 * pad || kw > in_w + 2 * pad)
            throw std::invalid_argument("Conv2D: cekirdek (padding dahil) girdiden buyuk");
    }

public:
    // Dikdortgen cekirdek
    Conv2DLayer(int in_c, int in_h, int in_w, int out_c, int kh, int kw, int stride, int pad)
        : in_c(in_c), in_h(in_h), in_w(in_w), out_c(out_c), kh(kh), kw(kw), stride(stride), pad(pad),
          out_h(0), out_w(0), K(0), P(0), Wm(1, 1), b(1, 1), dW(1, 1), db(1, 1), col(1, 1) {
        validate();
        out_h = (in_h + 2 * pad - kh) / stride + 1;
        out_w = (in_w + 2 * pad - kw) / stride + 1;
        K = in_c * kh * kw;
        P = out_h * out_w;
        Wm = Matrix::random(out_c, K, 0.0, std::sqrt(2.0 / K));   // He baslatma
        b  = Matrix::zeros(out_c, 1);
        dW = Matrix::zeros(out_c, K);
        db = Matrix::zeros(out_c, 1);
    }
    // Kare cekirdek
    Conv2DLayer(int in_c, int in_h, int in_w, int out_c, int kernel, int stride = 1, int pad = 0)
        : Conv2DLayer(in_c, in_h, in_w, out_c, kernel, kernel, stride, pad) {}

    int out_channels() const { return out_c; }
    int out_height()   const { return out_h; }
    int out_width()    const { return out_w; }
    int out_features() const { return out_c * P; }
    const Matrix& weights() const { return Wm; }

    void set_compute_input_grad(bool on) override { need_dx = on; }

    // x: (in_c*in_h*in_w x B)  ->  (out_c*out_h*out_w x B)
    Matrix forward(const Matrix& x) override {
        if (x.rows != in_c * in_h * in_w)
            throw std::invalid_argument("Conv2D: girdi satir sayisi " + std::to_string(x.rows)
                                        + ", beklenen " + std::to_string(in_c * in_h * in_w));
        batch = x.cols;
        const int B = batch;
        const size_t PB = (size_t)P * B;
        if (col.rows != K || col.cols != (int)PB) col = Matrix(K, (int)PB);

        // ---- im2col ----
        for (int c = 0; c < in_c; c++)
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++) {
                    const int k = (c * kh + ky) * kw + kx;
                    double* drow = col.data.data() + (size_t)k * PB;
                    for (int oh = 0; oh < out_h; oh++) {
                        const int ih = oh * stride + ky - pad;
                        for (int ow = 0; ow < out_w; ow++) {
                            const int iw = ow * stride + kx - pad;
                            double* dst = drow + (size_t)(oh * out_w + ow) * B;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w)
                                std::fill(dst, dst + B, 0.0);
                            else
                                std::memcpy(dst, x.data.data() + (size_t)((c * in_h + ih) * in_w + iw) * B,
                                            sizeof(double) * B);
                        }
                    }
                }

        // ---- GEMM: (out_c x K) * (K x P*B) ----
        Matrix Y = Wm * col;
        Matrix out(out_c * P, B, std::move(Y.data));   // bellek duzeni zaten ayni

        // ---- bias (kanal basina) ----
        for (int co = 0; co < out_c; co++) {
            const double bv = b.data[co];
            double* p = out.data.data() + (size_t)co * PB;
            for (size_t i = 0; i < PB; i++) p[i] += bv;
        }
        return out;
    }

    // grad_out: (out_c*P x B)  ->  grad_input: (in_c*in_h*in_w x B)
    Matrix backward(const Matrix& grad_out) override {
        const int B = batch;
        const size_t PB = (size_t)P * B;
        Matrix dYm(out_c, (int)PB, grad_out.data);       // ayni bellek duzeni

        for (int co = 0; co < out_c; co++) {             // db
            double s = 0.0;
            const double* p = dYm.data.data() + (size_t)co * PB;
            for (size_t i = 0; i < PB; i++) s += p[i];
            db.data[co] = s;
        }

        Matrix dYT = dYm.transpose();                    // (P*B x out_c)
        dW = (col * dYT).transpose();                    // (K x out_c) -> (out_c x K)

        if (!need_dx) return Matrix(1, 1);               // ilk katman: kullanilmayacak

        Matrix dcol = Wm.transpose() * dYm;              // (K x P*B)
        Matrix dx(in_c * in_h * in_w, B, 0.0);

        // ---- col2im: satir parcalarini geri topla ----
        for (int c = 0; c < in_c; c++)
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++) {
                    const int k = (c * kh + ky) * kw + kx;
                    const double* srow = dcol.data.data() + (size_t)k * PB;
                    for (int oh = 0; oh < out_h; oh++) {
                        const int ih = oh * stride + ky - pad;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int ow = 0; ow < out_w; ow++) {
                            const int iw = ow * stride + kx - pad;
                            if (iw < 0 || iw >= in_w) continue;
                            const double* src = srow + (size_t)(oh * out_w + ow) * B;
                            double* dst = dx.data.data() + (size_t)((c * in_h + ih) * in_w + iw) * B;
                            for (int j = 0; j < B; j++) dst[j] += src[j];
                        }
                    }
                }
        return dx;
    }

    void update(Optimizer* opt, const std::string& id) override {
        opt->step(Wm, dW, id + "_W");
        opt->step(b,  db, id + "_b");
    }

    void save(std::ostream& os) const override { write_matrix(os, Wm); write_matrix(os, b); }
    void load(std::istream& is) override {
        read_into(is, Wm, name() + ".W");
        read_into(is, b,  name() + ".b");
    }

    std::string name() const override {
        return "Conv2D(" + std::to_string(in_c) + "x" + std::to_string(in_h) + "x" + std::to_string(in_w)
             + "->" + std::to_string(out_c) + ",k" + std::to_string(kh) + "x" + std::to_string(kw)
             + ",s" + std::to_string(stride) + ",p" + std::to_string(pad) + ")";
    }
};

// ============================================================
//  MaxPool2D: her pencerenin en buyuk degerini alir.
//  Backward: gradyan sadece o en buyuk elemana gider (esitlikte ilk eleman).
//  Pencereler girdiye sigmiyorsa (floor modu) artik kenar satir/sutunlar kullanilmaz.
// ============================================================
class MaxPool2DLayer : public Layer {
    int ch, in_h, in_w, pool, stride, out_h, out_w;
    std::vector<int> argmax;   // her (cikti satiri, batch) icin kazanan GIRDI satiri
    int batch = 0;

public:
    MaxPool2DLayer(int channels, int in_h, int in_w, int pool = 2, int stride = 0)
        : ch(channels), in_h(in_h), in_w(in_w), pool(pool), stride(stride > 0 ? stride : pool),
          out_h(0), out_w(0) {
        if (channels <= 0 || in_h <= 0 || in_w <= 0) throw std::invalid_argument("MaxPool2D: boyutlar > 0 olmali");
        if (pool <= 0)                               throw std::invalid_argument("MaxPool2D: pool > 0 olmali");
        if (pool > in_h || pool > in_w)              throw std::invalid_argument("MaxPool2D: pencere girdiden buyuk");
        out_h = (in_h - pool) / this->stride + 1;
        out_w = (in_w - pool) / this->stride + 1;
    }

    int out_channels() const { return ch; }
    int out_height()   const { return out_h; }
    int out_width()    const { return out_w; }
    int out_features() const { return ch * out_h * out_w; }

    Matrix forward(const Matrix& x) override {
        if (x.rows != ch * in_h * in_w)
            throw std::invalid_argument("MaxPool2D: girdi satir sayisi " + std::to_string(x.rows)
                                        + ", beklenen " + std::to_string(ch * in_h * in_w));
        const int B = x.cols;
        batch = B;
        Matrix out(ch * out_h * out_w, B);
        argmax.assign((size_t)out.rows * B, 0);

        for (int c = 0; c < ch; c++)
            for (int oh = 0; oh < out_h; oh++)
                for (int ow = 0; ow < out_w; ow++) {
                    const int r_out = (c * out_h + oh) * out_w + ow;
                    double* o = out.data.data() + (size_t)r_out * B;
                    int*   ix = argmax.data() + (size_t)r_out * B;
                    bool first = true;
                    for (int dy = 0; dy < pool; dy++)
                        for (int dx = 0; dx < pool; dx++) {
                            const int r_in = (c * in_h + oh * stride + dy) * in_w + ow * stride + dx;
                            const double* src = x.data.data() + (size_t)r_in * B;
                            if (first) {
                                for (int j = 0; j < B; j++) { o[j] = src[j]; ix[j] = r_in; }
                                first = false;
                            } else {
                                // dallanmasiz (vektorlenebilir); kati '>' ile esitlikte ilk eleman kalir
                                for (int j = 0; j < B; j++) {
                                    const double v = src[j];
                                    const bool gt = v > o[j];
                                    o[j]  = gt ? v : o[j];
                                    ix[j] = gt ? r_in : ix[j];
                                }
                            }
                        }
                }
        return out;
    }

    Matrix backward(const Matrix& grad_out) override {
        const int B = batch;
        Matrix dx(ch * in_h * in_w, B, 0.0);
        const int rows_out = ch * out_h * out_w;
        for (int r = 0; r < rows_out; r++) {
            const double* g = grad_out.data.data() + (size_t)r * B;
            const int*   ix = argmax.data() + (size_t)r * B;
            for (int j = 0; j < B; j++)
                dx.data[(size_t)ix[j] * B + j] += g[j];
        }
        return dx;
    }

    std::string name() const override {
        return "MaxPool2D(" + std::to_string(ch) + "x" + std::to_string(in_h) + "x" + std::to_string(in_w)
             + ",k" + std::to_string(pool) + ",s" + std::to_string(stride) + ")";
    }
};
