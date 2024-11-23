import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

matplotlib.rcParams["lines.linewidth"] = 1.75 * matplotlib.rcParams["lines.linewidth"]
matplotlib.rcParams["lines.markersize"] = 1.75 * matplotlib.rcParams["lines.markersize"]
matplotlib.rcParams.update({"font.size": 1.75 * matplotlib.rcParams["font.size"]})

N = 72 * 256
K = 4 * N
bandwidth = 600 * 10**9
flops = 125 * 885 / 1695 * 10**12

def ideal(m, n, k):
    fp16_gb = 2 * m * k + 2 * k * n + 2 * m * n
    int4_gb = 2 * m * k + k * n / 2 + 2 * m * n + 2 * (k / 128) * n
    tf = 2 * m * n * k
    fp16_perf = max(fp16_gb / bandwidth, tf / flops)
    int4_perf = max(int4_gb / bandwidth, tf / flops)
    return fp16_perf / int4_perf

def ideal2(m, n, k):
    fp16_gb = 2 * m * k + 2 * k * n + 2 * m * n
    int4_gb = 2 * m * k + (k/2) * n / 2 + 2 * m * n + 2 * ((k/2) / 128) * n + 4 * n * (k / 2) / 16
    tf = 2 * m * n * (k/2)
    tf2 = 2 * m * n * k
    fp16_perf = max(fp16_gb / bandwidth, tf2 / flops)
    int4_perf = max(int4_gb / bandwidth, tf / flops)
    return fp16_perf / int4_perf

batchsizes = [1, 2, 4, 8, 16, 32, 64, 128] # , 256, 512, 1024, 2048]
expected = [ideal(m, N, K) for m in batchsizes]
expected2 = [ideal2(m, N, K) for m in batchsizes]

####
marlin_df = pd.read_csv("result/marlin_sustained.csv")
marlin=marlin_df[(marlin_df.model=="ideal")&(marlin_df.batch<=128)]["tot_q_sp"].tolist()
sparse_marlin=marlin_df[(marlin_df.model=="ideal")&(marlin_df.batch<=128)]["tot_q_2_4_sp"].tolist()

awq_df = pd.read_csv("result/awq_sustained.csv").astype({"batch": "int32"})
awq = awq_df[(awq_df.model=="ideal")&(awq_df.batch<=128)]["tot_q_sp"].tolist()

torch_df = pd.read_csv("result/torchao_sustained.csv")
torch = torch_df[(torch_df.M<=128)]["speedup"].tolist()

bitsandbytes_df = pd.read_csv("result/bitsandbytes_sustained.csv")
bitsandbytes = bitsandbytes_df[(bitsandbytes_df.model=="ideal")&(bitsandbytes_df.batch<=128)]["tot_q_sp"].tolist()

exllamav2_df = pd.read_csv("result/exllamav2_sustained.csv")
exllamav2 = exllamav2_df[(exllamav2_df.model=="ideal")&(exllamav2_df.batch<=128)]["tot_q_sp"].tolist()
####

plt.figure(figsize=(14.5, 7.5))
plt.grid()
plt.xticks(range(len(batchsizes)), [str(b) for b in batchsizes])

plt.plot(expected, "--", color="black", alpha=0.5)
plt.plot(expected, "o", color="black", label="Ideal dense")
plt.plot(expected2, "--", color="grey", alpha=0.5)
plt.plot(expected2, "o", color="grey", label="Ideal sparse")
plt.plot(marlin, "--", color="C0", alpha=0.5)
plt.plot(marlin, "o", color="C0", label="Marlin")
#plt.plot(sparse_marlin, "--", color="C2", alpha=0.5)
#plt.plot(sparse_marlin, "o", color="C2", label="Sparse-Marlin")
plt.plot(torch, "--", color="C1", alpha=0.5)
plt.plot(torch, "o", color="C1", label="torch-nightly")
plt.plot(exllamav2, "--", color="C5", alpha=0.5)
plt.plot(exllamav2, "o", color="C5", label="exllamav2")
plt.plot(awq, "--", color="C3", alpha=0.5)
plt.plot(awq, "o", color="C3", label="AWQ")
plt.plot(bitsandbytes, "--", color="C4", alpha=0.5)
plt.plot(bitsandbytes, "o", color="C4", label="bitsandbytes")

plt.ylim(None, 7.2)
plt.legend(loc="upper center", ncol=4)
plt.title("16bit$\\times$4bit (group=128) mul with 72k$\\times$18k matrix on NVIDIA A10 - LOCKED BASE CLOCK")
plt.xlabel("Batchsize")
plt.ylabel("Speedup over FP16 PyTorch (calling CUTLASS)")
plt.subplots_adjust(left=0.05, bottom=0.12, right=.99, top=0.9, wspace=0.2, hspace=0.2)
plt.savefig("result/sustained_smarlin.pdf", bbox_inches="tight")
plt.show()