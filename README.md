# Unofficial CUDA implementation of CoConv


use `pip3 install -r requirements.txt` to get all pre-requisites.</br>
Use command `python3 Evaluate.py` to get speedups and check correctness.</br>
Use a free gpu by explicitly setting which gpu to use with `CUDA_VISIBLE_DEVICES` environment variable.</br>
Ignore file doesn't exist errors. It is just for clean up reasons.

Cuda5 does not support pytorch so please test on 2,3,4.
Please be wary of the memory usage, we are testing with very large files in the Evaluate.py script. Both secondary storage and VRAM.
