"""
Lightweight OpenAI-compatible inference server using Unsloth + PEFT.
Drop-in replacement for vLLM when driver/version conflicts prevent vLLM.
Exposes /v1/completions and /health — same interface the eval shard expects.

Run in the rinlekha conda env (not rinlekha-serve):
  pip install fastapi uvicorn
  python serving/inference_server.py \
      --base-model unsloth/gemma-3-4b-it \
      --adapter outputs/run3-r16-lr1e4-ep5 \
      --port 8000
"""
import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from unsloth import FastLanguageModel

app = FastAPI()

model = None
tokenizer = None
model_name = "rinlekha"


# ── Request / response schemas (OpenAI-compatible) ────────────────────────────

class CompletionRequest(BaseModel):
    model: str = "rinlekha"
    prompt: str
    max_tokens: int = 900
    temperature: float = 0.1
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    stop: list[str] | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature > 0 else None,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][prompt_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Apply stop sequences
    finish_reason = "stop"
    if req.stop:
        for stop_seq in req.stop:
            if stop_seq in text:
                text = text[:text.index(stop_seq)]
                break
    else:
        finish_reason = "length" if len(generated_ids) >= req.max_tokens else "stop"

    completion_tokens = len(generated_ids)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        model=model_name,
        choices=[CompletionChoice(text=text.strip(), index=0, finish_reason=finish_reason)],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ── Startup ───────────────────────────────────────────────────────────────────

def load_model(base_model: str, adapter_path: str, max_seq_length: int = 2048) -> None:
    global model, tokenizer
    print(f"Loading base model: {base_model}")
    base, tok = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    print(f"Loading LoRA adapter: {adapter_path}")
    base = PeftModel.from_pretrained(base, adapter_path)
    FastLanguageModel.for_inference(base)
    model, tokenizer = base, tok
    print("Model ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",     default="unsloth/gemma-3-4b-it")
    parser.add_argument("--adapter",        default="outputs/run3-r16-lr1e4-ep5")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--host",           default="0.0.0.0")
    parser.add_argument("--port",           type=int, default=8000)
    args = parser.parse_args()

    load_model(args.base_model, args.adapter, args.max_seq_length)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
