# How to run
## Setup

Set environment variables: add the following statement to `~/.bashrc` file and execute the command `source ~/.bashrc` to reflush system environment variables.
```
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CPATH=/usr/local/cuda-12.1/include${CPATH:+:${CPATH}}
```
