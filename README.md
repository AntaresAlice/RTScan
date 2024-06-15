# RTScan: Efficient Scan with Ray Tracing Cores

<!-- ![RTScan](https://user-images.githubusercontent.com/46842476/205199552-4ab40b40-745a-4a9b-b19e-1838a2cd7662.png) -->


## Requirements
- NVCC. Tested on 12.1.105.
- CMake. Tested on 3.27.7.
- GCC/G++. Tested on 7.5.0.
- OptiX. Tested on 7.5. The project already includes OptiX SDK 7.5. 
A RTX-capable GPU (Turing architecture and later) from Nvidia. Tested on RTX 3090.

## Setup

Set environment variables: add the following statement to `~/.bashrc` file and execute the command `source ~/.bashrc` to reflush system environment variables.
```
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CPATH=/usr/local/cuda-12.1/include${CPATH:+:${CPATH}}
```

## Run experiments
```
$ cd .
$ mkdir optix-scan/build bin log data
$ bash script/gen_data.sh
$ python script/run.py
```