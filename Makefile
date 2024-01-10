rt-scan-build = ./optix-scan/build
optix-lib = ./optix-scan/build/lib

rt-scan-src = ./optix-scan/src/optixScan
rtscan-2c-src = ./optix-scan/src/rtscan_2c
rtscan-interval-spacing-src = ./optix-scan/src/rtscan_interval_spacing
rtc3-src = ./optix-scan/src/rtc3
rtc1-src = ./optix-scan/src/rtc1

rt-scan-libs = $(optix-lib)/liboptixScan.a $(optix-lib)/libsutil_7_sdk.so
rtscan-2c-libs = $(optix-lib)/librtscan_2c.a $(optix-lib)/libsutil_7_sdk.so
rtscan-interval-spacing-libs = $(optix-lib)/librtscan_interval_spacing.a $(optix-lib)/libsutil_7_sdk.so
rtc3-libs = $(optix-lib)/librtc3.a $(optix-lib)/libsutil_7_sdk.so
rtc1-libs = $(optix-lib)/librtc1.a $(optix-lib)/libsutil_7_sdk.so

rt-scan-srcs = $(rt-scan-src)/optixScan.cpp $(rt-scan-src)/optixScan.cu $(rt-scan-src)/optixScan.h $(rt-scan-src)/state.h $(rt-scan-src)/aabb.cu
rtscan-2c-srcs = $(rtscan-2c-src)/optixScan.cpp $(rtscan-2c-src)/optixScan.cu $(rtscan-2c-src)/optixScan.h $(rtscan-2c-src)/state.h $(rtscan-2c-src)/aabb.cu
rtscan-interval-spacing-srcs = $(rtscan-interval-spacing-src)/optixScan.cpp $(rtscan-interval-spacing-src)/optixScan.cu $(rtscan-interval-spacing-src)/optixScan.h $(rtscan-interval-spacing-src)/state.h $(rtscan-interval-spacing-src)/aabb.cu
rtc3-srcs = $(rtc3-src)/optixScan.cpp $(rtc3-src)/optixScan.cu $(rtc3-src)/optixScan.h $(rtc3-src)/state.h $(rtc3-src)/aabb.cu
rtc1-srcs = $(rtc1-src)/optixScan.cpp $(rtc1-src)/optixScan.cu $(rtc1-src)/optixScan.h $(rtc1-src)/state.h $(rtc1-src)/aabb.cu

LDFLAGS = -L. $(rt-scan-libs) -ldl -pthread -fopenmp -lcudart -lcuda -Wl,-rpath='$(optix-lib)'
LDFLAGS-rtc3 = -L. $(rtc3-libs) -ldl -pthread -fopenmp -lcudart -lcuda -Wl,-rpath='$(optix-lib)'
LDFLAGS-rtc1 = -L. $(rtc1-libs) -ldl -pthread -fopenmp -lcudart -lcuda -Wl,-rpath='$(optix-lib)'
LDFLAGS-rtscan-2c = -L. $(rtscan-2c-libs) -ldl -pthread -fopenmp -lcudart -lcuda -Wl,-rpath='$(optix-lib)'
LDFLAGS-rtscan-interval-spacing = -L. $(rtscan-interval-spacing-libs) -ldl -pthread -fopenmp -lcudart -lcuda -Wl,-rpath='$(optix-lib)'


ifndef DATA_N
	DATA_N = 1e8
endif

ifndef VAREA_N
	VAREA_N = 128
endif

ifndef ENCODE
	ENCODE = 0
endif

ifndef PRIMITIVE_TYPE
	PRIMITIVE_TYPE = 1
endif

ifndef SMALL_DATA_RANGE
	SMALL_DATA_RANGE = 0
endif

ifndef TPCH
	TPCH = 0
endif

ifndef BUILD_TYPE
	BUILD_TYPE = Release
endif

ifeq ($(BUILD_TYPE), Debug)
	GPLUS = -g
else
	GPLUS = 
endif

rtscan: rtscan.cpp remap.cpp rt.h ./bin/librt_cuda.a timer.h $(optix-lib)/liboptixScan.a
	g++ -std=c++11 $^ -o ./bin/$@ $(LDFLAGS) -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N) -DENCODE=$(ENCODE) -DTPCH=$(TPCH) -DVAREA_N=$(VAREA_N) -DDISTRIBUTION=$(DISTRIBUTION) $(GPLUS)

rtscan_2c: rtscan_2c.cpp rt.h ./bin/librt_cuda.a timer.h $(optix-lib)/librtscan_2c.a
	g++ -std=c++11 $^ -o ./bin/$@ $(LDFLAGS-rtscan-2c) -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N)

rtscan_interval_spacing: rtscan_interval_spacing.cpp rt.h timer.h $(optix-lib)/librtscan_interval_spacing.a
	g++ -std=c++11 $^ -o ./bin/$@ $(LDFLAGS-rtscan-interval-spacing) -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N)

bindex_cuda: bindex_cuda.cpp bindex.h ./bin/libcuda_and.a timer.h
	g++ -std=c++11 $^ -o ./bin/$@ -ldl -pthread -fopenmp -lcudart -lcuda -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N) -DONLY_REFINE=$(ONLY_REFINE) -DONLY_DATA_SIEVING=$(ONLY_DATA_SIEVING)

bindex: bindex.cpp
	g++ -std=c++11 $^ -o ./bin/$@ -pthread -mavx2 -march=native -fopenmp -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N) -DVAREA_N=$(VAREA_N)

rtc3: rtc3.cpp remap.cpp rt.h ./bin/librt_cuda.a timer.h $(optix-lib)/librtc3.a
	g++ -std=c++11 $^ -o ./bin/$@ $(LDFLAGS-rtc3) -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N) -DENCODE=$(ENCODE) -DDISTRIBUTION=$(DISTRIBUTION) $(GPLUS)

rtc1: rtc1.cpp rt.h ./bin/librt_cuda.a timer.h $(optix-lib)/librtc1.a
	g++ -std=c++11 $^ -o ./bin/$@ $(LDFLAGS-rtc1) -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N) -DDISTRIBUTION=$(DISTRIBUTION) $(GPLUS)

./bin/rt_cuda.o: rt_cuda.cu
	nvcc -c $^ -o $@ -DDATA_N=$(DATA_N) $(GPLUS)

./bin/librt_cuda.a: ./bin/rt_cuda.o
	ar cr $@ $^

./bin/cuda_and.o: cuda_and.cu
	nvcc -c $^ -o $@ -DDATA_N=$(DATA_N) $(GPLUS)

./bin/libcuda_and.a: ./bin/cuda_and.o
	ar cr $@ $^

$(optix-lib)/libsutil_7_sdk.so $(optix-lib)/liboptixScan.a $(optix-lib)/librtc3.a $(optix-lib)/librtc1.a $(optix-lib)/librtscan_2c.a $(optix-lib)/librtscan_interval_spacing.a: \
$(rt-scan-srcs) $(rtc3-srcs) $(rtc1-srcs) $(rtscan-2c-srcs) $(rtscan-interval-spacing-srcs)
	cd $(rt-scan-build) && \
	cmake ../src/ -D CMAKE_C_COMPILER=/usr/bin/gcc-7 -D CMAKE_BUILD_TYPE=$(BUILD_TYPE) -D DEBUG_ISHIT_CMP_RAY=$(DEBUG_ISHIT_CMP_RAY) -D DEBUG_INFO=$(DEBUG_INFO) -D PRIMITIVE_TYPE=$(PRIMITIVE_TYPE) -D SMALL_DATA_RANGE=$(SMALL_DATA_RANGE) && \
	make

clean:
	rm -rf ./bin/* optix-scan/build/*