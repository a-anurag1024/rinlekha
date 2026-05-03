"""
ARCHIVED — superseded by llama-cpp-python server (serving/start_server.sh).

Why this was replaced:
  This file went through several iterations trying to serve a QLoRA adapter
  directly from Python. Each approach hit a different wall:

  1. FastAPI + Unsloth: deadlocked inside anyio's run_in_threadpool because
     Unsloth's compiled Gemma3 forward pass can't run inside asyncio threads.
     Workaround: switched to Flask (threaded=False).

  2. Flask + Unsloth for_inference(): torch.compile recompiles for every
     autoregressive step because the attention mask grows by 1 each step
     ([1,24], [1,25], ...). Each recompile stalls on CPU for ~3 seconds.
     Result: ~0.6 tok/s (measured: 162s for 50 tokens).

  3. Flask + TORCHDYNAMO_DISABLE=1: disables compile, eliminating the stall.
     But Unsloth Zoo's eager-mode patches have per-op Python overhead; each
     forward pass takes ~473ms instead of the ~50-70ms expected from GPU
     bandwidth math. Result: ~1.3 tok/s — 10-20x below expected.

  Root cause: Unsloth Zoo's patches are designed assuming torch.compile fuses
  them into CUDA kernels. In eager mode they become individual Python→CUDA
  dispatches with kernel-launch overhead per operation per layer.

New approach (no workarounds):
  - scripts/merge_and_export.py  — merges adapter into base model (fp16 HF)
  - scripts/convert_to_gguf.sh   — converts to GGUF Q8_0 via llama.cpp tools
  - serving/start_server.sh      — starts llama-cpp-python server
    Exposes the same /v1/completions and /health endpoints.
    Expected speed on RTX 4050: 10-20 tok/s.
"""
