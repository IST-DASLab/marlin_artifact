import argparse
import sys
import time

from datasets import load_dataset
import torch

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-path', type=str, help='path to the model checkpoint folder (HuggingFace format)',
    )
    parser.add_argument(
        '--n-gpus', type=int, default=1, help='number of GPUs'
    )
    parser.add_argument(
        '--batch-size-list', nargs='*', type=int, default=[1, 2, 4, 8, 16, 32, 64, 128], help='list of evaluated batch sizes',
    )
    parser.add_argument(
        '--n-in-tokens', type=int, default=64, help='number of input tokens',
    )
    parser.add_argument(
        '--n-out-tokens', type=int, default=64, help='number of output tokens',
    )
    parser.add_argument(
        '--n-warmup-reps', type=int, default=5, help='number of warm up iterations',
    )
    parser.add_argument(
        '--n-reps', type=int, default=10, help='number of iterations after warm up (disabled if --min-runtime is non-negative)',
    )
    parser.add_argument(
        '--min-runtime', type=float, default=100., help='minimum run time after warm up, set to negative to disable',
    )
    parser.add_argument(
        '--vllm-gpu-memory-utilization', type=float, default=.9, help='ratio of GPU memory reserved for vLLM (decrease to prevent cuda oom caused by temporary tensors; increase to reserve more kv cache for larger batch sizes)',
    )
    parser.add_argument(
        '--vllm-enforce-eager', type=str2bool, default=False, help='whether to force vLLM to use eager mode (False: use CUDA Graph; True: not to use CUDA Graph, saving GPU memory but running slowly)',
    )
    args = parser.parse_args()
    return args


def get_wikitext2(split: str = 'train') -> str:
    dataset = load_dataset(path='wikitext', name='wikitext-2-raw-v1', split=split)
    dataset_str = '\n\n'.join(dataset['text'])
    return dataset_str


def main():
    print(sys.argv)
    args = parse_args()
    print(args)

    model_path: str = args.model_path
    tensor_parallel_size: int = args.n_gpus
    batch_size_list: list[int] = args.batch_size_list
    n_in_tokens: int = args.n_in_tokens
    n_out_tokens: int = args.n_out_tokens
    n_warmup_reps: int = args.n_warmup_reps
    n_reps: int = args.n_reps
    min_runtime: float = args.min_runtime
    vllm_gpu_memory_utilization: float = args.vllm_gpu_memory_utilization
    vllm_enforce_eager: bool = args.vllm_enforce_eager

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,  # decrease to prevent cuda oom; increase to reserve more kv cache
        swap_space=0,
        enforce_eager=vllm_enforce_eager,
        max_seq_len_to_capture=32768,
        disable_custom_all_reduce=True,
        max_model_len=n_in_tokens + n_out_tokens,
    )

    tokenizer = llm.get_tokenizer()
    data_str: str = get_wikitext2()
    encoding_full: list[int] = tokenizer.encode(text=data_str, add_special_tokens=False)

    all_metrics: list[dict] = []
    for batch_size in batch_size_list:
        llm.llm_engine.scheduler_config.max_model_len = n_in_tokens + n_out_tokens - 1  # max seq len, default 2048
        llm.llm_engine.scheduler_config.max_num_batched_tokens = 32768  # kv cache numbers default 2048
        llm.llm_engine.scheduler_config.max_num_seqs = batch_size  # max batch size, default 256
        llm.llm_engine.scheduler_config.max_paddings = 0  # default 256
        # TODO vllm/entrypoints/llm.py _run_engine(): add assert len(step_outputs) == num_requests

        indices = torch.randint(len(encoding_full) - n_in_tokens + 1, [batch_size]).tolist()
        prompt_token_ids: list[list[int]] = [encoding_full[i: i + n_in_tokens] for i in indices]

        sampling_params = SamplingParams(
            n=1,
            temperature=.8,
            top_p=.95,
            ignore_eos=True,
            max_tokens=n_out_tokens,
        )

        time_full: list[float] = []
        time_first: list[float] = []

        count_reps = 0
        start_time_0 = None
        if min_runtime >= 0.:
            n_reps = 1000

        for i in range(n_warmup_reps + n_reps):
            count_reps = i + 1
            if i == n_warmup_reps:
                start_time_0 = time.time()
            start_time = time.time()
            outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=True)
            end_time = time.time()
            first_token_time = outputs[0].metrics.first_token_time
            time_full.append(end_time - start_time)
            time_first.append(first_token_time - start_time)
            if i >= n_warmup_reps and 0. <= min_runtime <= end_time - start_time_0:
                break

        metrics = {
            'model_path': model_path,
            'n_gpus': tensor_parallel_size,
            'batch_size': batch_size,
            'n_in_tokens': n_in_tokens,
            'n_out_tokens': n_out_tokens,
            'n_warmup_reps': n_warmup_reps,
            'n_reps': count_reps - n_warmup_reps,
            'vllm-gpu-memory-utilization': vllm_gpu_memory_utilization,
            'vllm-enforce-eager': vllm_enforce_eager,
            'time_full': time_full,
            'time_first': time_first,
        }
        all_metrics.append(metrics)
        print(metrics)

    print(f'\nALL METRICS:\n')
    print(all_metrics)


if __name__ == '__main__':
    main()
