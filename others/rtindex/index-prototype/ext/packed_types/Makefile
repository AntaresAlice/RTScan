INCDIRS := include

CC := g++
NVCC := nvcc
STD := c++14
OPT := 3
CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
NVCCGENCODE = -arch=sm_35
NVCCFLAGS := -O$(OPT) -std=$(STD) $(NVCCGENCODE) -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS)) --expt-extended-lambda

INCPARAMS := $(addprefix -I, $(INCDIRS))

all: example

example: example.cu include/packed_types.cuh | bin
	$(NVCC) $(NVCCFLAGS) $(INCPARAMS) example.cu -o bin/example

debug: OPT := 0
debug: CCFLAGS := -O$(OPT) -std=$(STD) -Wall -Wextra -fopenmp
debug: XCCFLAGS := $(addprefix -Xcompiler ,$(CCFLAGS))
debug: NVCCFLAGS := -O$(OPT) -std=$(STD) -ccbin $(CC) $(XCCFLAGS) $(NVCCGENCODE) --expt-extended-lambda -g -Xptxas -v -UNDEBUG -DDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -UNDEBUG
profile: all

clean:
	$(RM) -r bin

bin:
	@mkdir -p $@

.PHONY: clean all bin