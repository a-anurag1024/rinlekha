"""
Run on Google Colab with a T4 GPU runtime.
Paste each section into its own cell.

Bypasses Unsloth's save_pretrained_merged/save_pretrained_gguf entirely —
both paths silently produce the base model instead of the merged weights.
Uses standard PEFT merge_and_unload() instead, which modifies weights in-place
(no full-model memory copy, safe on T4's 15 GB VRAM).

Prerequisite: set HF_TOKEN in Colab Secrets (key icon → New secret →
  Name: HF_TOKEN → Value: your HF write-access token).
You must also have accepted the Gemma licence at
  https://huggingface.co/google/gemma-3-4b-it
"""


# ── CELL 1: install ──────────────────────────────────────────────────────────
# Run once. Restart runtime when Colab prompts, then continue to Cell 2.

# !pip install -q transformers peft accelerate safetensors "torchao>=0.16.0"
# !pip install -q gguf sentencepiece protobuf


# ── CELL 2: merge + convert ───────────────────────────────────────────────────
# Pure PEFT merge — no Unsloth involved.

import gc, os, subprocess, torch
from google.colab import userdata
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN   = userdata.get("HF_TOKEN")
BASE_MODEL = "google/gemma-3-4b-it"
ADAPTER    = "a-anurag1024/rinlekha-gemma3-4b-finetuned"
MERGED_DIR = "/content/rinlekha-merged"
LLAMA_DIR  = "/content/llama.cpp"
GGUF_PATH  = "/content/rinlekha-q8.gguf"

# 1 — load base model in fp16 on GPU (~8 GB VRAM, fits T4)
print(f"Loading base model: {BASE_MODEL}")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
    token=HF_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

# 2 — attach LoRA adapter
print(f"Attaching adapter: {ADAPTER}")
peft_model = PeftModel.from_pretrained(base, ADAPTER, token=HF_TOKEN)

# 3 — merge in-place (PEFT modifies each layer's weight tensor directly,
#     so peak VRAM stays at the base model size, not 2×)
print("Merging LoRA weights in-place...")
merged = peft_model.merge_and_unload()

# 4 — save merged fp16 model
print(f"Saving to {MERGED_DIR} ...")
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

# free VRAM before the conversion step
del merged, peft_model, base
gc.collect()
torch.cuda.empty_cache()
print("Merge complete.")

# 5 — clone llama.cpp and convert to GGUF Q8_0
if not os.path.exists(LLAMA_DIR):
    print("Cloning llama.cpp...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ggerganov/llama.cpp", LLAMA_DIR],
        check=True,
    )

# install converter deps (gguf package may not be present in base Colab)
subprocess.run(["pip", "install", "-q", "gguf", "sentencepiece", "protobuf"], check=True)

print("Converting to GGUF Q8_0...")
result = subprocess.run(
    ["python", f"{LLAMA_DIR}/convert_hf_to_gguf.py",
     MERGED_DIR, "--outtype", "q8_0", "--outfile", GGUF_PATH],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("STDOUT:", result.stdout[-3000:])
    print("STDERR:", result.stderr[-3000:])
    raise RuntimeError("GGUF conversion failed — see output above")
print(result.stdout[-1000:])

size_gb = os.path.getsize(GGUF_PATH) / 1e9
print(f"\nGGUF ready: {GGUF_PATH}  ({size_gb:.1f} GB)")


# ── CELL 2b: inspect Gemma3Model.set_vocab in Colab's llama.cpp ──────────────
# Shows what the current code looks like so we know exactly what to patch.

import re

CONVERTER = "/content/llama.cpp/convert_hf_to_gguf.py"

with open(CONVERTER) as f:
    src = f.read()

m = re.search(
    r"(class Gemma3Model\b.*?def set_vocab\(self\):)(.*?)(\n    def |\nclass )",
    src, re.DOTALL,
)
if m:
    print("Current Gemma3Model.set_vocab body:")
    print(m.group(2))
else:
    print("ERROR: could not locate Gemma3Model.set_vocab — check the class name")


# ── CELL 2c: copy tokenizer.model + retry conversion ─────────────────────────
# Colab's llama.cpp already has the correct set_vocab with the tokenizer.model
# check. The problem is that AutoTokenizer.save_pretrained only writes the fast
# tokenizer JSON — it does NOT copy the SentencePiece tokenizer.model file.
# Without it the converter falls back to _set_vocab_gpt2() which fails on the
# Gemma 3 hash. Fix: explicitly download tokenizer.model from the adapter repo.

import os, shutil, subprocess
from google.colab import userdata
from huggingface_hub import hf_hub_download

HF_TOKEN   = userdata.get("HF_TOKEN")
ADAPTER    = "a-anurag1024/rinlekha-gemma3-4b-finetuned"
MERGED_DIR = "/content/rinlekha-merged"
CONVERTER  = "/content/llama.cpp/convert_hf_to_gguf.py"
GGUF_PATH  = "/content/rinlekha-q8.gguf"

# Download tokenizer.model from the adapter and copy it into the merged dir
print("Downloading tokenizer.model from adapter repo...")
tok_model_src = hf_hub_download(
    repo_id=ADAPTER, filename="tokenizer.model", token=HF_TOKEN,
)
shutil.copy(tok_model_src, os.path.join(MERGED_DIR, "tokenizer.model"))
print(f"tokenizer.model present: {os.path.isfile(os.path.join(MERGED_DIR, 'tokenizer.model'))}")

print("Retrying GGUF conversion...")
result = subprocess.run(
    ["python", CONVERTER, MERGED_DIR, "--outtype", "q8_0", "--outfile", GGUF_PATH],
    capture_output=True, text=True,
)
print("STDOUT:", result.stdout[-2000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("GGUF conversion failed — see output above")

print(f"\nGGUF ready: {GGUF_PATH}  ({os.path.getsize(GGUF_PATH)/1e9:.1f} GB)")


# ── CELL 2d: quick inference test ────────────────────────────────────────────
# Loads the GGUF on CPU (no server needed) and generates ~150 tokens.
# A correctly merged model will start with "## APPLICANT SUMMARY".
# If it starts with numbered/bolded headers or a letterhead, the merge failed.

import subprocess
# Install pre-built CPU wheel (~30s) — avoids compiling from source
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall",
     "llama-cpp-python",
     "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"],
    check=True,
)

from llama_cpp import Llama

GGUF_PATH   = "/content/rinlekha-q8.gguf"
INSTRUCTION = (
    "You are a senior credit analyst at an Indian NBFC. "
    "Write a structured credit memo for the borrower profile "
    "below following institutional format exactly."
)
SAMPLE_INPUT = """\
=== BORROWER PROFILE ===

DEMOGRAPHICS
  Age             : 34 years
  City Tier       : Tier 2

EMPLOYMENT
  Type            : Salaried Private
  Monthly Income  : ₹45,000
  Tenure          : 3.5 years

CREDIT PROFILE
  CIBIL Score     : 720
  Missed Payments (last 24m): 1
  Existing EMI    : ₹8,000/month

LOAN REQUEST
  Amount          : ₹3,00,000
  Tenure          : 36 months

DERIVED METRICS
  FOIR (post-loan): 40.6%  [policy ceiling: 55%]"""

prompt = (
    f"### Instruction:\n{INSTRUCTION}\n\n"
    f"### Input:\n{SAMPLE_INPUT}\n\n"
    f"### Response:\n"
)

print("Loading GGUF on CPU (no GPU needed for this test)...")
# llama-cpp-python calls fileno() on stderr during init, which Jupyter's
# wrapped streams don't support — swap in the real streams temporarily.
import sys
_out, _err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
try:
    llm = Llama(model_path=GGUF_PATH, n_gpu_layers=0, n_ctx=2048, verbose=False)
finally:
    sys.stdout, sys.stderr = _out, _err

print("Generating (150 tokens)...\n")
out = llm(prompt, max_tokens=150, temperature=0.1, stop=["### Instruction:"])
text = out["choices"][0]["text"].strip()

print("=" * 60)
print(text[:600])
print("=" * 60)
print(f"\nStarts with '## APPLICANT SUMMARY': {text.startswith('## APPLICANT SUMMARY')}")


# ── CELL 3: upload GGUF to HF Hub ────────────────────────────────────────────

from huggingface_hub import HfApi

GGUF_REPO = "a-anurag1024/rinlekha-gguf"
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=GGUF_REPO, repo_type="model", exist_ok=True)

print(f"Uploading to {GGUF_REPO} ...")
api.upload_file(
    path_or_fileobj=GGUF_PATH,
    path_in_repo="rinlekha-q8.gguf",
    repo_id=GGUF_REPO,
    repo_type="model",
)
print(f"\nDone. Download on your laptop:")
print(f"  huggingface-cli download {GGUF_REPO} rinlekha-q8.gguf --local-dir outputs/")