import os
import time

from matplotlib import pyplot as plt
import numpy as np
import torch

import marlin


def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res


def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    torch.cuda.synchronize()
    return A, B, C, s, workspace


def benchmark_problems():
    gpu = torch.cuda.get_device_name(0)
    if 'A100' in gpu:
        sms = 108
    elif 'A10' in gpu:
        sms = 72
    elif '3090' in gpu:
        sms = 82
    elif 'A6000' in gpu:
        sms = 84
    else:
        sms = -1

    batch_sizes = 2 ** np.arange(17)
    layer_sizes = [(2 ** i,) * 2 for i in range(12, 16)]

    metrics = {}
    for layer in layer_sizes:
        metrics[layer] = []
        for batch in batch_sizes:
            A, B, C, s, workspace = get_problem(batch, layer[1], layer[0], 128)
            res = benchmark(lambda: marlin.mul(A, B, C, s, workspace, -1, -1, sms))
            metrics[layer].append(res)
    return metrics


def plot_roofline(metrics):
    def get_ops(b: int, n: int, m: int):
      return 2 * b * m * n + 2 * m * n


    def get_bits(b: int, n: int, m: int):
      return 16 * b * n + 4 * m * n + 16 * b * m + 16 * m * n // 128


    def get_intensity(b: int, n: int, m: int):
      return get_ops(b, m, n) / get_bits(b, m, n)

    def get_performance(b: int, m: int, n: int):
      return get_ops(b, m, n) / np.array(metrics[(n, m)])[np.log2(b).astype(int)]

    batch_sizes = 2 ** np.arange(17)

    peak_performance = 125e12  # 125 TFLOPS/s
    peak_bandwidth = 600e9  # 600 GB/s

    min_performance = peak_performance
    max_intensity = 0.

    plt.figure(figsize=(8., 4.))
    for m in 2 ** np.arange(12, 16):
      n = m
      intensity = get_intensity(batch_sizes, m, n) * 8.
      performance = get_performance(batch_sizes, m, n)
      min_performance = min(np.min(performance), min_performance)
      max_intensity = max(np.max(intensity), max_intensity)
      plt.plot(intensity, performance * 1e-12, marker='.', label=rf'${m}\times{n}$', zorder=20)
    plt.plot(
        [.8 * min_performance / peak_bandwidth, peak_performance / peak_bandwidth],
        [.8 * min_performance * 1e-12, peak_performance * 1e-12],
        color='k', linestyle='-',
    )
    base_performance = (885e6 / 1695e6) * peak_performance
    plt.plot(
        [.8 * min_performance / peak_bandwidth, peak_performance / peak_bandwidth],
        [.8 * min_performance * 1e-12, peak_performance * 1e-12],
        color='k', linestyle='-',
    )
    plt.plot(
        [base_performance / peak_bandwidth, 1.5 * max_intensity],
        [base_performance * 1e-12, base_performance * 1e-12],
        color='k', linestyle='-',
    )
    plt.plot(
        [peak_performance / peak_bandwidth, 1.5 * max_intensity],
        [peak_performance * 1e-12, peak_performance * 1e-12],
        color='k', linestyle='-',
    )
    plt.plot(
        [.8 * min_performance / peak_bandwidth, peak_performance / peak_bandwidth],
        [peak_performance * 1e-12, peak_performance * 1e-12],
        color='k', linestyle='--', linewidth=1.,
    )
    plt.plot(
        [.8 * min_performance / peak_bandwidth, base_performance / peak_bandwidth],
        [base_performance * 1e-12, base_performance * 1e-12],
        color='k', linestyle='--', linewidth=1.,
    )
    plt.plot(
        [peak_performance / peak_bandwidth, peak_performance / peak_bandwidth],
        [.8 * min_performance * 1e-12, peak_performance * 1e-12],
        color='k', linestyle='--', linewidth=1.,
    )
    plt.plot(
        [base_performance / peak_bandwidth, base_performance / peak_bandwidth],
        [.8 * min_performance * 1e-12, base_performance * 1e-12],
        color='k', linestyle='--', linewidth=1.,
    )
    plt.xscale('log')
    plt.xlim(2.5e12 / peak_bandwidth, 1.5 * max_intensity)
    plt.xticks([10, 100, 108.8, 208.3, 1000, 10000], ['10', '', '108.8', '208.3', '1000', '10000'])
    plt.xlabel('Arithmetic Intensity [FLOP/Byte]')
    plt.yscale('log')
    plt.ylim(2.5)  # .8 * min_performance * 1e-12
    plt.yticks([2.5, 5, 10, 20, 40, 65.3, 80, 100, 125, 160], ['2.5', '5', '10', '20', '40', '65.3', '80', '100', '125', '160'])
    plt.ylabel('Performance [TFLOP/s]')
    plt.text(11, 3.5, 'Memory-Bound', c='gray')
    plt.text(220, 69, 'Compute-Bound', c='gray')
    plt.text(2100, 84, 'Thermal Throttling', c='gray')
    plt.text(5, 129, 'Max Performance (Boost Clock)', fontstyle='italic')
    plt.text(5, 67, 'Max Performance (Base Clock)', fontstyle='italic')
    plt.title(r'MARLIN Roofline Analysis with Batch Size $2^{0}$-$2^{16}$ on NVIDIA A10')
    plt.tick_params(axis='both', which='both', length=0)
    plt.grid()
    plt.legend(title='Weight Matrix Shape', framealpha=1.)
    plt.gca().set_facecolor((1., 1., 1., 1.))
    plt.gcf().set_facecolor((1., 1., 1., 0.))
    plt.tight_layout()
    plt.savefig(os.path.join('result', 'marlin_roofline.pdf'), bbox_inches='tight', pad_inches=.02, transparent=False)


if __name__ == '__main__':
    metrics = benchmark_problems()
    print('marlin roofline metrics:')
    print(metrics)
    plot_roofline(metrics)
