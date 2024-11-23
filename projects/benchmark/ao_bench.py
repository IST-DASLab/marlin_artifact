try:
    import hqq
except ImportError:
    raise "triton and hqq required to run this benchmark"

from io import StringIO

import pandas as pd
import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from torchao.prototype.hqq import pack_2xint4
from torchao.prototype.hqq.hqq_tinygemm_linear import HQQLinearTorchWeightOnlyInt4

from triton.testing import do_bench

import time

BASE_QUANT_CONFIG = {
    "optimize": True,
    "view_as_float": False,
    "nbits": 4,
    "bitpack": False,
    "axis": 1,
}

def benchmark(f, warmup=1, iter=100):
    for i in range(warmup + iter):
        f()
        if i == warmup:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

def bench_hqq(x, hqq_linear: HQQLinear | HQQLinearTorchWeightOnlyInt4, transposed=False, tinygemm=False):
    def reference_fn():
        W_dq = hqq_linear.dequantize()
        _ = x @ W_dq.T if not transposed else x @ W_dq
    fn = reference_fn if not tinygemm else lambda: hqq_linear(x)

    #t = benchmark(fn)
    t = do_bench(fn)
    return t


def run_benchmark(
    shape, group_size, dtype, axis=1, transposed=False, quant_dtype=torch.uint8
):
    qcfg = {
        **BASE_QUANT_CONFIG,
        **dict(group_size=group_size, axis=axis),
    }
    M, N, K = shape

    x = (
        torch.randn(M, K, dtype=dtype, device="cuda")
        if not transposed
        else torch.randn(M, N, dtype=dtype, device="cuda")
    )
    linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cpu")

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )
    quant_config.update({"weight_quant_params": qcfg})

    """ hqq_linear = HQQLinear(linear, quant_config, compute_dtype=dtype, del_orig=False)

    # Reference
    # ...

    # Custom kernel
    W_q, meta = hqq_linear.W_q, hqq_linear.meta
    scales, zeros = meta["scale"], meta["zero"]

    W_q = (
        W_q.reshape(meta["shape"])
        if quant_config["weight_quant_params"]["bitpack"] == False
        else W_q
    )
    W_q = W_q.to(dtype=quant_dtype)
    scales = scales.reshape(N, -1)
    zeros = zeros.reshape(N, -1) """

    should_run_tinygemm = dtype == torch.bfloat16 and not transposed
    #should_run_tinygemm = True
    if should_run_tinygemm:
        _ = quant_config["weight_quant_params"].pop("bitpack")

        #print(linear.weight.device)#, quant_config.device)
        hqq_int4mm = HQQLinearTorchWeightOnlyInt4(
            linear, quant_config, compute_dtype=dtype, del_orig=False
        ).to("cuda")
        #int4_time = bench_hqq(x, hqq_int4mm, transposed=transposed, tinygemm=True)
        #int4_time = bench_hqq(x, hqq_int4mm, transposed=transposed, tinygemm=True)
        int4_time = benchmark(lambda: hqq_int4mm(x))
        #print(int4_time)

    # Reference
    linear = linear.to("cuda")
    #ref_time = bench_hqq(x, hqq_linear, transposed=transposed)
    #ref_time = bench_hqq(x, hqq_linear, transposed=transposed, tinygemm=True)
    ref_time = benchmark(lambda: linear(x))
    #del linear
    #print(ref_time)

    """ print(f"{shape=}, {group_size=}, {dtype=}, {transposed=}:")

    print(
        f"Ref: {ref_time:.4f}ms",
        f"Torch int4mm: {int4_time:.4f}ms" if should_run_tinygemm else "",
        f"SpeedUp: {ref_time/int4_time:.4f}",
    )
    print() """
    return (
        ref_time,
        int4_time if should_run_tinygemm else -1,
        ref_time/int4_time
    )


SHAPES = [
    [16, 4096, 4096],
    [32, 4096, 4096],
    [128, 4096, 4096],
    [256, 4096, 4096],
    [512, 4096, 4096],
    [1024, 4096, 4096],
]

DTYPES = [torch.bfloat16] #[torch.float16, torch.bfloat16]
GROUP_SIZES = [128]
TRANSPOSED = [False] #[False, True]

HEADERS = [
    "M",
    "N",
    "K",
    "group_size",
    "dtype",
    "transposed",
    "ref",
    "tinygemm",
    "speedup"
]
data = []

N = 72 * 256
K = 72 * 256 * 4

torch.cuda.empty_cache()

if __name__ == "__main__":
    #print(torch.cuda.get_device_properties(0))

    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        shape = (M,N,K)
        for group_size in GROUP_SIZES:
            for dtype in DTYPES:
                for transposed in TRANSPOSED:
                    timings = run_benchmark(
                        shape, group_size, dtype, transposed=transposed
                    )
                    data.append((*shape, group_size, dtype, transposed, *timings))

    output = StringIO()
    df = pd.DataFrame(data, columns=HEADERS)
    df.to_csv(output, index=False)
    print(output.getvalue())
    # df.to_csv("benchmark_hqq_tinygemm.csv", index=False)