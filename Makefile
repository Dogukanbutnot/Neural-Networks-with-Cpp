CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# make OMP=1  -> OpenMP ile paralel matris carpimi
# make NOSIMD=1 -> SIMD cekirdeklerini kapat
ifeq ($(OMP),1)
CXXFLAGS += -fopenmp
endif
ifeq ($(NOSIMD),1)
CXXFLAGS += -DNN_NO_SIMD
endif

HDRS = matrix.h layer.h optimizer.h network.h dataloader.h mnist.h serialization.h scheduler.h earlystopping.h conv.h

all: nn bench sched_test opt_test grad_test es_test conv_test

nn: main.cpp matrix.cpp network.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ main.cpp matrix.cpp network.cpp

bench: bench.cpp matrix.cpp matrix.h
	$(CXX) $(CXXFLAGS) -o $@ bench.cpp matrix.cpp

sched_test: scheduler_test.cpp matrix.cpp network.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ scheduler_test.cpp matrix.cpp network.cpp

opt_test: optimizer_test.cpp matrix.cpp network.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ optimizer_test.cpp matrix.cpp network.cpp

grad_test: gradcheck_test.cpp matrix.cpp network.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ gradcheck_test.cpp matrix.cpp network.cpp

es_test: earlystop_test.cpp matrix.cpp network.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ earlystop_test.cpp matrix.cpp network.cpp

conv_test: conv_test.cpp matrix.cpp network.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ conv_test.cpp matrix.cpp network.cpp

# Tum testleri calistir
test: sched_test opt_test grad_test es_test conv_test
	./sched_test && ./opt_test && ./grad_test && ./es_test && ./conv_test

clean:
	rm -f nn bench sched_test opt_test grad_test es_test conv_test
