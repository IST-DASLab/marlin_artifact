import sys

import numpy as np
import torch

import time

def benchmark(f, warmup=10, iter=100):
    for i in range(warmup + iter):
        f()
        if i == warmup:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    return (time.time() - tick) / iter

def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B, C, s, thread_k, thread_n, sms):
    from awq.quantize.qmodule import WQLinear
    l = WQLinear(4, 128, A.shape[1], C.shape[1], False, torch.device('cuda:0'))
    res = benchmark(lambda: l(A))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = {
    'ideal': [
        (4 * 256 * SMS, 256 * SMS)
    ],
    '7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    '13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    '33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    '70B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ]
}

print("groupsize,model,batch,tot_q_s,tot_q_sp")
for groupsize in [128]:
    #print('groupsize=%d' % groupsize)
    #print()
    for model, layers in MODELS.items():
        #print(model)
        for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]: # , 2048]:
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0}
            for layer in layers:
                A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], groupsize)
                res_d = benchmark_dense(A, B_ref, C)
                if model == 'ideal' and batch == 16:
                    res_q = benchmark_quant(A, B, C, s, 64, 256, SMS)
                else:
                    res_q = benchmark_quant(A, B, C, s, -1, -1, SMS)
                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            """ print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup']
            )) """
            print(str(groupsize)+',%s,%04d,%.5f,%.2f' % (model, batch, tot_q['s'], tot_q['speedup']))
        #print()
