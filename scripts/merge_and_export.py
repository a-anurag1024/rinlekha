"""
Export the fine-tuned adapter directly to GGUF Q8_0 using Unsloth's
save_pretrained_gguf. This is a single step (no intermediate HF merge).

Previous approach (save_pretrained_merged → convert_to_gguf.sh) silently
produced the BASE model — Unsloth's post-save validation threw
"# of LoRAs = 319 != # saved modules = 0", confirming zero LoRA weights
were merged into the saved files. The base GGUF was then served instead
of the fine-tuned model.

save_pretrained_gguf drives llama.cpp's converter directly without going
through the broken Unsloth module-merge path, so it correctly applies the
LoRA weights.

Usage:
  python scripts/merge_and_export.py
  python scripts/merge_and_export.py \
      --adapter outputs/run3-r16-lr1e4-ep5 \
      --output  outputs/rinlekha-q8.gguf

After this script completes:
  bash serving/start_server.sh outputs/rinlekha-q8.gguf
"""
import argparse
from pathlib import Path

from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/run3-r16-lr1e4-ep5",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--output", default="outputs/rinlekha-q8.gguf",
                        help="Output GGUF file path (without .gguf extension — "
                             "Unsloth appends it automatically)")
    args = parser.parse_args()

    # Strip .gguf suffix if the user included it — Unsloth appends it itself
    output_stem = str(args.output)
    if output_stem.endswith(".gguf"):
        output_stem = output_stem[:-5]

    print(f"Loading adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print(f"Exporting to GGUF Q8_0: {output_stem}.gguf")
    model.save_pretrained_gguf(output_stem, tokenizer, quantization_method="q8_0")

    output_path = Path(output_stem + ".gguf")
    if output_path.exists():
        size_gb = output_path.stat().st_size / 1e9
        print(f"\nGGUF written: {output_path}  ({size_gb:.1f} GB)")
    else:
        print(f"\nWARNING: expected output not found at {output_path}")
        print("Check for Unsloth errors above.")

    print("\nNext: bash serving/start_server.sh outputs/rinlekha-q8.gguf")


if __name__ == "__main__":
    main()