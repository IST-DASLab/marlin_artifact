import numpy as np
import time
import torch

# from awq.quantize.qmodule import WQLinear
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV

def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

DEV = torch.device('cuda:1')
N = 72 * 256
K = 72 * 256 * 4

B = torch.randn((K, N), dtype=torch.half, device=DEV)

for M in [1, 2, 4, 8, 16, 32, 64, 128]:
    A = torch.randn((M, K), dtype=torch.half, device=DEV)
    # l = WQLinear_GEMM(4, 128, K, N, False, torch.device('cuda:0'))
    l = WQLinear_GEMM(4, 128, K, N//8, False, torch.device('cuda:1'))
    # l = WQLinear_GEMV(4, 128, K, N, False, torch.device('cuda:0'))
    torch.cuda.synchronize()
    fp16 = benchmark(lambda: torch.matmul(A, B))
    int4 = benchmark(lambda: l(A))
    print(M, fp16 / int4)