# https://github.com/neuralmagic/nm-vllm/blob/9daca33a6fdc429802f448e1ea71630c996c9740/benchmarks/benchmark_serving.py

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import random
import os
import sys
import time
import traceback
from typing import AsyncGenerator
import warnings

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.
    ttft: float = 0.  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    prompt_len: int = 0
    error: str = ""


async def request_func(
    request_func_input: RequestFuncInput,
    pbar: tqdm = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            'min_tokens': request_func_input.output_len,
            'ignore_eos': True,
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=request_func_input.api_url, json=payload, headers={}) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk == "[DONE]":
                            output.latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)
                            # NOTE: Some completion API might have a last usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                output.generated_text += data["choices"][0]["text"]

                    if len(output.generated_text) > 0:
                        output.success = True
                    else:
                        output.success = False
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def sample_requests(
        num_requests: int,
        input_len: int,
        output_len: int,
        tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, int, int]]:
    with open(os.path.join('e2e', 'sonnet.txt'), 'r') as f:
        dataset_str = ''.join(f.readlines())
    encoding_full: list[int] = tokenizer.encode(text=dataset_str, add_special_tokens=False)
    indices = np.random.randint(0, len(encoding_full) - input_len + 1, [num_requests])
    prompt_token_ids: list[list[int]] = [encoding_full[i: i + input_len] for i in indices]
    prompts = tokenizer.batch_decode(prompt_token_ids)
    sampled_requests: list[tuple[str, int, int]] = [(p, input_len, output_len) for p in prompts]
    return sampled_requests


async def get_request(
        input_requests: list[tuple[str, int, int]],
        request_rate: float,
) -> AsyncGenerator[tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1. / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float


def calculate_metrics(
        input_requests: list[tuple[str, int, int]],
        outputs: list[RequestFuncOutput],
        dur_s: float,
        tokenizer: PreTrainedTokenizerBase,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note: this may inflate the output token count slightly
            output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            completed += 1
        else:
            actual_output_lens.append(0)

    print('actual_output_lens', actual_output_lens)
    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
        api_url: str,
        model_id: str,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: list[tuple[str, int, int]],
        request_rate: float,
        disable_tqdm: bool,
):
    # print("Starting initial single prompt test run...")
    # test_prompt, test_prompt_len, test_output_len = input_requests[0]
    # test_input = RequestFuncInput(
    #     model=model_id,
    #     prompt=test_prompt,
    #     api_url=api_url,
    #     prompt_len=test_prompt_len,
    #     output_len=test_output_len,
    # )
    # test_output = await request_func(request_func_input=test_input)
    # if not test_output.success:
    #     raise ValueError(
    #         "Initial test run failed - Please make sure benchmark arguments "
    #         f"are correctly specified. Error: {test_output.error}")
    # else:
    #     print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )
        tasks.append(asyncio.create_task(request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s='Inter-token Latency (incl. 1st token)', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result


def main():
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    api_url = f"http://{args.host}:{args.port}/v1/completions"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    input_requests = sample_requests(
        num_requests=args.num_prompts,
        input_len=args.sonnet_input_len,
        output_len=args.sonnet_output_len,
        tokenizer=tokenizer,
    )

    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
        ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--sonnet-input-len",
        type=int,
        default=64,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-output-len",
        type=int,
        default=64,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
             "then all the requests are sent at time 0. "
             "Otherwise, we use Poisson process to synthesize "
             "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
