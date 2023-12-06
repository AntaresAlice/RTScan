all: example

example: example.cu cuda_helpers.cuh
	nvcc -O3 -std=c++11 -arch=sm_35 --expt-extended-lambda example.cu -o example

clean:
	rm -rf example
