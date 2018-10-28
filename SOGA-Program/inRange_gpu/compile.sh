#!/bin/bash

nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libinRange_gpu.so --shared inRange_gpu.cu
sudo cp inRange_gpu.h /usr/local/include/
sudo cp libinRange_gpu.so /usr/local/lib/

exit 0
