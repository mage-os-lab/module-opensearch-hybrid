from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT / "data/cache/models/neural-sparse/doc-v3-distill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the official sparse TorchScript runtime")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = importlib.import_module("torch")
    tokenizers = importlib.import_module("tokenizers")
    tokenizer: Any = tokenizers.Tokenizer.from_file(str(MODEL_DIRECTORY / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding()
    texts = ["coffee table", "trail running shoe"] * ((args.batch_size + 1) // 2)
    encoded = tokenizer.encode_batch(texts[: args.batch_size])
    inputs = {
        "input_ids": torch.tensor(
            [item.ids for item in encoded],
            dtype=torch.long,
            device=args.device,
        ),
        "attention_mask": torch.tensor(
            [item.attention_mask for item in encoded],
            dtype=torch.long,
            device=args.device,
        ),
    }
    model = torch.jit.load(
        str(MODEL_DIRECTORY / "opensearch-neural-sparse-encoding-doc-v3-distill.pt"),
        map_location="cpu",
    ).eval().float().to(args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(inputs)["output"]
    elapsed = time.perf_counter() - started
    print(
        f"{args.device} sparse probe passed: shape={tuple(output.shape)}, "
        f"nonzero={int(torch.count_nonzero(output).item())}, seconds={elapsed:.4f}"
    )


if __name__ == "__main__":
    main()
