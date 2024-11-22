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
flops = 125 * 10**12

def ideal(m, n, k):
    fp16_gb = 2 * m * k + 2 * k * n + 2 * m * n
    int4_gb = 2 * m * k + k * n / 2 + 2 * m * n + 2 * (k / 128) * n
    tf = 2 * m * n * k
    fp16_perf = max(fp16_gb / bandwidth, tf / flops)
    int4_perf = max(int4_gb / bandwidth, tf / flops)
    return fp16_perf / int4_perf

def ideal2(m, n, k):
    fp16_gb = 2 * m * k + 2 * k * n + 2 * m * n
    int4_gb = 2 * m * k +  (k/2) * n / 2 +  2 * m * n +  2 * ((k/2) / 128) * n +  2 * n * k / 16
    tf = 2 * m * n * (k/2)
    tf2 = 2 * m * n * k
    fp16_perf = max(fp16_gb / bandwidth, tf2 / flops)
    int4_perf = max(int4_gb / bandwidth, tf / flops)
    return fp16_perf / int4_perf

batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
expected = [ideal(m, N, K) for m in batchsizes]
expected2 = [ideal2(m, N, K) for m in batchsizes]

####
marlin_df = pd.read_csv("result/marlin.csv")
marlin=marlin_df[(marlin_df.model=="ideal")&(marlin_df.batch<=128)]["tot_q_sp"].tolist()
sparse_marlin=marlin_df[(marlin_df.model=="ideal")&(marlin_df.batch<=128)]["tot_q_2_4_sp"].tolist()

awq_df = pd.read_csv("result/awq.csv").astype({"batch": "int32"})
awq = awq_df[(awq_df.model=="ideal")&(awq_df.batch<=128)]["tot_q_sp"].tolist()

torch_df = pd.read_csv("result/torchao.csv")
torch = torch_df[(torch_df.M<=128)]["speedup"].tolist()

bitsandbytes_df = pd.read_csv("result/bitsandbytes.csv")
bitsandbytes = bitsandbytes_df[(bitsandbytes_df.model=="ideal")&(bitsandbytes_df.batch<=128)]["tot_q_sp"].tolist()

exllamav2_df = pd.read_csv("result/exllamav2.csv")
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
plt.plot(sparse_marlin, "--", color="C2", alpha=0.5)
plt.plot(sparse_marlin, "o", color="C2", label="Sparse-Marlin")
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
plt.title("16bit$\\times$4bit (group=128) mul with 72k$\\times$18k matrix on NVIDIA A10")
plt.xlabel("Batchsize")
plt.ylabel("Speedup over FP16 PyTorch (calling CUTLASS)")
plt.subplots_adjust(left=0.05, bottom=0.12, right=.99, top=0.9, wspace=0.2, hspace=0.2)
plt.savefig("result/peak_smarlin.pdf", dpi=50, bbox_inches="tight")
plt.show()

models = ["Llama7B", "Llama13B", "Llama33B", "Llama65B", "Falcon180B"]
#a10 = [3.69, 3.79, 3.83, 3.85, 3.74]
marlin_df = pd.read_csv("result/marlin.csv")
a10=[]
for model in models:
    a10.append( marlin_df[(marlin_df.model==model)]["tot_q_sp"].values[0] )
#rtx3090 = [3.60, 3.69, 4.14, 4.05, 4.25]
#a6000 = [3.40, 3.65, 3.78, 3.86, 3.80]
#a100 = [2.22, 2.80, 3.04, 3.20, 3.64]

plt.figure(figsize=(17.5, 7.5))
plt.grid()
#plt.xticks([1.5, 6.5, 11.5, 16.5, 21.5], models)
plt.xticks([0, 1, 2, 3, 4], models)
plt.bar([i + 0 for i in range(5)], a10, width=0.5, label="NVIDIA A10")
#plt.bar([5 * i + 1 for i in range(5)], rtx3090, width=1, label="NVIDIA 3090")
#plt.bar([5 * i + 2 for i in range(5)], a6000, width=1, label="NVIDIA A6000")
#plt.bar([5 * i + 3 for i in range(5)], a100, width=1, label="NVIDIA A100")
plt.ylim(None, 5)
plt.legend(loc="upper center", ncol=6)
plt.title("Marlin (group=128) performance on layer shapes of popular models - BATCHSIZE 16")
plt.ylabel("Speedup over FP16 PyTorch (calling CUTLASS)")
plt.savefig("result/models.pdf", bbox_inches="tight")
plt.show()