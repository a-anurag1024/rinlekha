# ── WSL stability: keep WSL alive during long training runs ───────────────
# Run this ONCE before starting training (in any terminal):
#   powercfg /change standby-timeout-ac 0   (Windows: disable sleep on AC)
#   powercfg /change monitor-timeout-ac 0   (Windows: disable monitor sleep)
# Restore after training:
#   powercfg /change standby-timeout-ac 30
#   powercfg /change monitor-timeout-ac 15

# ── Install tmux if not present ───────────────────────────────────────────
# sudo apt-get install tmux -y

# ── Dry-run: confirm model loads and fits in VRAM ─────────────────────────
python -c "
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/gemma-3-4b-it',
    max_seq_length=2048,
    load_in_4bit=True,
)
used = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'Model loaded. VRAM: {used:.2f} / {total:.2f} GB ({used/total*100:.0f}%)')
" 2>&1 | grep -v "patch\|faster\|finetuning"
