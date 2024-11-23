import numpy as np
import time
import torch

import bitsandbytes as bnb

# def benchmark(f, warmup=10, iter=100):
def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1)
    return res

DEV = torch.device('cuda:0')

N = 72 * 256
K = 72 * 256 * 4

B_ref = torch.randn((K, N), dtype=torch.half, device=DEV)

model="ideal"
groupsize=128
print("groupsize,model,batch,tot_q_s,tot_q_sp")
for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
#for M in [512, 1024, 2048]:
    A = torch.randn((M, K), dtype=torch.half, device=DEV)
    C = torch.randn((M, N), dtype=torch.half, device=DEV)

    l = bnb.nn.Linear4bit(
        K, N, bias=False, compute_dtype=torch.float16, compress_statistics=False
    )
    l.weight.blocksize = 128
    l.cuda()
    torch.cuda.synchronize()

    fp16 = benchmark(lambda: torch.matmul(A, B_ref))
    int4 = benchmark(lambda: l(A))
    #print(M, fp16 / int4)
    print(str(groupsize)+',%s,%04d,%.5f,%.2f' % (model, M, int4, fp16/int4))
