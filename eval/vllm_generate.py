#!/usr/bin/env python3
"""
Generate outputs from a vLLM OpenAI-compatible server for prompts stored in a Parquet file.

Input:
  - Parquet at /path/to/prompts.parquet
  - Column: `prompt` (string)

Output:
  - JSONL at <output_dir>/<out_name>.jsonl
  - Each line includes BOTH keys for compatibility across scripts:
      {"input": <prompt>, "prompt": <prompt>, "output": <generated_text>}
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Iterable, List, Optional, Tuple


def _repo_root() -> str:
    # This file lives in <repo>/eval/.
    return str(__import__("pathlib").Path(__file__).resolve().parents[1])


# Default to a repo-local, writable output directory.
DEFAULT_OUTPUT_DIR = __import__("os").path.join(_repo_root(), "eval", "output")


def _chunks(xs: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def read_prompts_from_parquet(path: str, column: str) -> List[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: pyarrow is required to read parquet. "
            "Install it in your env: `python -m pip install pyarrow`."
        ) from e

    table = pq.read_table(path, columns=[column])
    if column not in table.column_names:
        raise ValueError(f"Column '{column}' not found. Available columns: {table.column_names}")
    prompts = table[column].to_pylist()
    # Defensive: coerce to str and filter Nones
    out: List[str] = []
    for i, p in enumerate(prompts):
        if p is None:
            continue
        if not isinstance(p, str):
            p = str(p)
        out.append(p)
    return out


def make_client(host: str, port: int, api_key: str):
    # openai>=1.0 client
    from openai import OpenAI  # type: ignore

    base_url = f"http://{host}:{port}/v1"
    return OpenAI(base_url=base_url, api_key=api_key)

def make_async_client(host: str, port: int, api_key: str):
    from openai import AsyncOpenAI  # type: ignore

    base_url = f"http://{host}:{port}/v1"
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def wait_for_server_ready(host: str, port: int, timeout_s: float = 300.0, poll_s: float = 0.5, server_proc: Optional[subprocess.Popen] = None) -> None:
    """Wait until GET /v1/models responds."""
    import requests  # type: ignore

    url = f"http://{host}:{port}/v1/models"
    t0 = time.time()
    last_err: Optional[str] = None
    while time.time() - t0 < timeout_s:
        if server_proc is not None and server_proc.poll() is not None:
            rc = server_proc.returncode
            raise RuntimeError(
                f"vLLM server process exited early (returncode={rc}) while waiting for readiness at {url}. "
                "Check the server logs above for the root cause (common: CUDA OOM, wrong --tp vs GPUs, bad model path)."
            )
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for vLLM server at {url}. Last error: {last_err}")


def start_vllm_openai_server(
    model_path: str,
    served_model_name: str,
    bind_host: str,
    port: int,
    max_model_len: int,
    tp: int,
    enforce_eager: bool,
    compilation_mode: str,
    cudagraph_mode: str,
    cuda_visible_devices: Optional[str],
    disable_torch_compile: bool,
    chat_template: Optional[str] = None,
) -> subprocess.Popen:
    """
    Start vLLM OpenAI server as a subprocess.
    Returns Popen handle; caller is responsible for termination.
    """
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        bind_host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--compilation-config",
        json.dumps({"mode": compilation_mode, "cudagraph_mode": cudagraph_mode}),
    ]
    if tp and tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if chat_template:
        cmd += ["--chat-template", chat_template]

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # vLLM 0.13 decorates some helper functions with @torch.compile
    # (e.g., vocab_parallel_embedding.get_masked_input_and_mask). On many HPC
    # systems, Triton/Inductor compilation fails due to missing Python.h.
    # Disabling Dynamo/Inductor prevents those codepaths from compiling.
    if disable_torch_compile:
        env.setdefault("TORCHDYNAMO_DISABLE", "1")
        env.setdefault("TORCHINDUCTOR_DISABLE", "1")

    # Inherit stdout/stderr so logs are visible.
    return subprocess.Popen(cmd, env=env)


async def _gen_one_async(
    client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Optional[List[str]],
    request_timeout: float,
    sem: asyncio.Semaphore,
) -> str:
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            timeout=request_timeout,
        )
        return resp.choices[0].message.content or ""


async def generate_all_async(
    host: str,
    port: int,
    api_key: str,
    model: str,
    prompts: List[str],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Optional[List[str]],
    request_timeout: float,
    concurrency: int,
    pbar,
) -> List[str]:
    client = make_async_client(host, port, api_key)
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def runner(i: int, p: str) -> Tuple[int, str]:
        out = await _gen_one_async(
            client=client,
            model=model,
            prompt=p,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            request_timeout=request_timeout,
            sem=sem,
        )
        if pbar is not None:
            pbar.update(1)
        return i, out

    tasks = [asyncio.create_task(runner(i, p)) for i, p in enumerate(prompts)]
    results: List[str] = [""] * len(prompts)
    for fut in asyncio.as_completed(tasks):
        i, out = await fut
        results[i] = out
    await client.close()
    return results


def generate_batch_chat(
    client,
    model: str,
    prompts: List[str],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Optional[List[str]],
    request_timeout: float,
):
    # vLLM OpenAI server supports the OpenAI Chat Completions API.
    # We send each prompt as a single-user message.
    outputs: List[str] = []
    for prompt in prompts:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            timeout=request_timeout,
        )
        outputs.append(resp.choices[0].message.content or "")
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name as exposed by the vLLM OpenAI server")
    ap.add_argument("--host", default="127.0.0.1", help="vLLM server host for client requests (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, required=True, help="vLLM server port (e.g. 8000)")
    ap.add_argument("--api_key", default="EMPTY", help="API key for OpenAI client (vLLM typically ignores it)")

    ap.add_argument("--input_parquet", required=True, help="Input parquet path")
    ap.add_argument("--prompt_column", default="prompt", help="Parquet column containing prompts (default: prompt)")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write jsonl output")
    ap.add_argument("--out_name", default=None, help="Output file base name (default: model_port_timestamp)")

    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for iteration (default: 8)")
    ap.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of prompts")
    ap.add_argument(
        "--take_last",
        type=int,
        default=None,
        help="If set, only generate on the last N prompts from the parquet (applied before --start/--limit).",
    )

    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p (default: 1.0)")
    ap.add_argument("--max_tokens", type=int, default=512, help="Max new tokens (default: 512)")
    ap.add_argument("--stop", nargs="*", default=None, help="Optional stop strings (space-separated)")

    ap.add_argument("--request_timeout", type=float, default=600.0, help="Per-request timeout seconds (default: 600)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (default: 0)")
    ap.add_argument("--print_every", type=int, default=50, help="Progress print frequency (default: 50)")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Async client concurrency (number of in-flight requests). Use >1 to speed up. (default: 1)",
    )
    ap.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Disable tqdm progress bar (fallback to periodic prints).",
    )
    ap.add_argument(
        "--start_server",
        action="store_true",
        help="Start a vLLM OpenAI server as a subprocess before generation, then stop it on exit.",
    )
    ap.add_argument(
        "--server_model_path",
        default=None,
        help="Model path for starting the server (defaults to --model).",
    )
    ap.add_argument(
        "--served_model_name",
        default=None,
        help="served-model-name for the server (defaults to --model).",
    )
    ap.add_argument(
        "--server_bind_host",
        default="0.0.0.0",
        help="Server bind host when --start_server is set (default: 0.0.0.0).",
    )
    ap.add_argument(
        "--server_ready_host",
        default="127.0.0.1",
        help="Host to poll for readiness when --start_server is set (default: 127.0.0.1).",
    )
    ap.add_argument(
        "--server_ready_timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for server readiness when --start_server is set (default: 600).",
    )
    ap.add_argument(
        "--server_max_model_len",
        type=int,
        default=2048,
        help="--max-model-len for vLLM server when --start_server is set (default: 2048).",
    )
    ap.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM server (multi-GPU) when --start_server is set (default: 1)",
    )
    ap.add_argument(
        "--server_enforce_eager",
        action="store_true",
        help="Pass --enforce-eager to vLLM server when --start_server is set.",
    )
    ap.add_argument(
        "--server_compilation_mode",
        default="NONE",
        help="Compilation mode for vLLM server (NONE/STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE/VLLM_COMPILE). Default: NONE",
    )
    ap.add_argument(
        "--server_cudagraph_mode",
        default="NONE",
        help="CUDAGraph mode for vLLM server (NONE/PIECEWISE/FULL/...). Default: NONE",
    )
    ap.add_argument(
        "--server_cuda_visible_devices",
        default=None,
        help="If set, export CUDA_VISIBLE_DEVICES for the server subprocess (e.g. '0,1').",
    )
    ap.add_argument(
        "--server_allow_torch_compile",
        action="store_true",
        help="Allow torch.compile inside vLLM server (default: disabled to avoid Python.h/Triton build issues on clusters).",
    )
    ap.add_argument(
        "--server_chat_template",
        default=None,
        help="Path to a chat template file or a template string for vLLM server.",
    )
    args = ap.parse_args()

    prompts = read_prompts_from_parquet(args.input_parquet, args.prompt_column)
    if args.take_last is not None:
        n = int(args.take_last)
        if n <= 0:
            raise ValueError("--take_last must be a positive integer")
        prompts = prompts[-n:]
    if args.start < 0 or args.start >= len(prompts):
        raise ValueError(f"--start out of range: {args.start} (num_prompts={len(prompts)})")
    prompts = prompts[args.start :]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_name = args.out_name or f"{args.model.replace('/', '_')}_p{args.port}_{ts}"
    out_path = os.path.join(args.output_dir, f"{out_name}.jsonl")

    server_proc: Optional[subprocess.Popen] = None
    if args.start_server:
        model_path = args.server_model_path or args.model
        served_name = args.served_model_name or args.model
        server_proc = start_vllm_openai_server(
            model_path=model_path,
            served_model_name=served_name,
            bind_host=args.server_bind_host,
            port=int(args.port),
            max_model_len=int(args.server_max_model_len),
            tp=int(args.tp),
            enforce_eager=bool(args.server_enforce_eager),
            compilation_mode=str(args.server_compilation_mode),
            cudagraph_mode=str(args.server_cudagraph_mode),
            cuda_visible_devices=args.server_cuda_visible_devices,
            disable_torch_compile=(not bool(args.server_allow_torch_compile)),
            chat_template=args.server_chat_template,
        )

        def _cleanup():
            if server_proc is None:
                return
            if server_proc.poll() is None:
                try:
                    server_proc.send_signal(signal.SIGTERM)
                    server_proc.wait(timeout=10)
                except Exception:
                    try:
                        server_proc.kill()
                    except Exception:
                        pass

        atexit.register(_cleanup)
        wait_for_server_ready(args.server_ready_host, int(args.port), timeout_s=float(args.server_ready_timeout), server_proc=server_proc)

    client = make_client(args.host, args.port, args.api_key)

    total = len(prompts)
    written = 0
    # Optional tqdm progress bar.
    pbar = None
    if not args.no_progress_bar:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=total, desc="Generating", unit="prompt", dynamic_ncols=True)
        except Exception:
            pbar = None

    if int(args.concurrency) > 1:
        outs = asyncio.run(
            generate_all_async(
                host=args.host,
                port=int(args.port),
                api_key=args.api_key,
                model=args.model,
                prompts=prompts,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                top_p=float(args.top_p),
                stop=args.stop,
                request_timeout=float(args.request_timeout),
                concurrency=int(args.concurrency),
                pbar=pbar,
            )
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for inp, out in zip(prompts, outs):
                f.write(json.dumps({"input": inp, "prompt": inp, "output": out}, ensure_ascii=False) + "\n")
        written = total
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            for batch_idx, batch_prompts in enumerate(_chunks(prompts, args.batch_size)):
                batch_out = generate_batch_chat(
                    client=client,
                    model=args.model,
                    prompts=batch_prompts,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    stop=args.stop,
                    request_timeout=args.request_timeout,
                )
                for inp, out in zip(batch_prompts, batch_out):
                    f.write(json.dumps({"input": inp, "prompt": inp, "output": out}, ensure_ascii=False) + "\n")
                    written += 1
                    if pbar is not None:
                        pbar.update(1)

                if args.sleep > 0:
                    time.sleep(args.sleep)

                if pbar is None and (written % args.print_every == 0 or written == total):
                    print(f"[vllm_generate] wrote {written}/{total} -> {out_path}", file=sys.stderr)
    if pbar is not None:
        pbar.close()

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



