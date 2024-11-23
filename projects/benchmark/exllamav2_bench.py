import numpy as np
import time
import torch
import torch.nn as nn

import exllamav2
from conversion.qparams import QParams
from conversion.adaptivegptq import AdaptiveGPTQ

def benchmark(f, warmup=10, iter=100):
# def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        if i == warmup:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

DEV = torch.device('cuda:0')

N = 72 * 256
K = 72 * 256 * 4

B_ref = torch.randn((K, N), dtype=torch.half, device=DEV)
with torch.inference_mode():
    DIV = 4
    K //= DIV
    qparams = QParams(128, [4], [1.], 4)
    l = nn.Linear(K, N, bias=False, dtype=torch.half, device=DEV)
    gptq = AdaptiveGPTQ(l)
    gptq.configure(
        group_size=qparams.group_size,
        bits=qparams.bits,
        bits_prop=qparams.bits_prop,
        scale_bits=qparams.scale_bits
    )
    gptq.hessian_inv = torch.eye(K, device=DEV)
    gptq.perm = torch.arange(K, device=DEV)
    gptq.perm_cpu = gptq.perm.cpu()
    gptq.quantize(keep_qweight=True)
    packed = gptq.pack('l', qparams)
    packed = {k.replace('l.', ''): v for k, v in packed.items()}

    K *= DIV
    packed['q_invperm'] = torch.arange(K, device=DEV)
    packed['q_perm'] = torch.arange(K, device=DEV)
    packed['q_scale'] = packed['q_scale'].repeat(DIV, 1)
    packed['q_scale_max'] = packed['q_scale_max'].repeat(DIV)
    packed['q_weight'] = packed['q_weight'].repeat(DIV, 1)
    scratch = torch.empty(K * N + 128, dtype=torch.half, device=DEV)
    scratch2 = scratch
    B = exllamav2.ext.make_q_matrix(packed, scratch)

    time.sleep(2.)

model="ideal"
groupsize=128
print("groupsize,model,batch,tot_q_s,tot_q_sp")
for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    A = torch.randn((M, K), dtype=torch.half, device=DEV)
    C = torch.randn((M, N), dtype=torch.half, device=DEV)
    fp16 = benchmark(lambda: torch.matmul(A, B_ref, out=C))
    int4 = benchmark(lambda: exllamav2.ext.exllamav2_ext.gemm_half_q_half(A, B, C, False))
    #print(fp16 / int4)
    print(str(groupsize)+',%s,%04d,%.5f,%.2f' % (model, M, int4, fp16/int4))
