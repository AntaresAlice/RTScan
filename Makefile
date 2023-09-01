# rt scan lib and src location
rt-scan-build = ./optix-scan/build

rt-scan-lib = ./optix-scan/build/lib

rt-scan-src = ./optix-scan/src/optixScan

rt-scan-libs = $(rt-scan-lib)/liboptixScan.a $(rt-scan-lib)/libsutil_7_sdk.so

rt-scan-srcs = $(rt-scan-src)/optixScan.cpp $(rt-scan-src)/optixScan.cu $(rt-scan-src)/optixScan.h $(rt-scan-src)/state.h $(rt-scan-src)/aabb.cu

LDFLAGS = -L. $(rt-scan-libs) -ldl -pthread -fopenmp -lcudart -lcuda -Wl,-rpath='$(rt-scan-lib)'

ifndef DATA_N
	DATA_N = 1e8
endif

ifndef VAREA_N
	VAREA_N = 128
endif

ifndef ENCODE
	ENCODE = 0
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

bindex: bindex.cpp remap.cpp bindex.h libcuda_and.a timer.h $(rt-scan-lib)/liboptixScan.a
	g++ -std=c++11 $^ -o $@ $(LDFLAGS) -mavx2 -march=native -DD_GLIBCXX_PARALLEL -DDATA_N=$(DATA_N) -DENCODE=$(ENCODE) -DTPCH=$(TPCH) -DVAREA_N=$(VAREA_N) -DDISTRIBUTION=$(DISTRIBUTION) $(GPLUS)

cuda_and.o: cuda_and.cu
	nvcc -c cuda_and.cu -o cuda_and.o -DDATA_N=$(DATA_N) $(GPLUS)

libcuda_and.a: cuda_and.o
	ar cr libcuda_and.a cuda_and.o

$(rt-scan-libs):$(rt-scan-srcs)
	cd $(rt-scan-build) && \
	cmake ../src/ -D CMAKE_C_COMPILER=/usr/bin/gcc-7 -D CMAKE_BUILD_TYPE=$(BUILD_TYPE) -D DEBUG_ISHIT_CMP_RAY=$(DEBUG_ISHIT_CMP_RAY) -D DEBUG_INFO=$(DEBUG_INFO) && \
	make

query_encoding: ./test/query_encoding.cpp bindex.h remap.cpp
	g++ -std=c++11 $^ -o $@ -ldl -pthread

clean:
	rm -rf bindex cuda_and.o libcuda_and.a optix-scan/build/*

# TODO: clean all optix linked libraries

clean-bindex:
	rm -rf bindex cuda_and.o libcuda_and.a

clean-rtscan:
	rm -rf $(rt-scan-lib)