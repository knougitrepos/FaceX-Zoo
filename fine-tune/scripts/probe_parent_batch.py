"""Bounded parent batch-size comparison, each invocation uses a fresh CUDA process."""
import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.dataset import FaceImageDataset
from modules.parent_batch_probe import measure_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True, choices=[64, 256])
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        loader = None
        if args.train_list:
            sample = set(random.Random(8972).sample(range(3141890), 4096))
            with args.train_list.open(encoding="utf-8") as stream:
                lines = [line for i, line in enumerate(stream) if i in sample]
            if len(lines) != 4096:
                raise ValueError("Incomplete VGGFace2 train list")
            random.Random(8972).shuffle(lines)
            index = Path(temp) / "sample.txt"
            index.write_text("".join(lines), encoding="utf-8")
            for line in lines:
                Path(line.rsplit(" ", 1)[0]).read_bytes()
            dataset = FaceImageDataset(str(args.train_list.parent), str(index), is_train=True)
            loader = DataLoader(dataset, batch_size=args.batch, num_workers=4, pin_memory=True,
                                prefetch_factor=4)
        result = measure_batch(args.batch, loader)
        if loader is not None:
            result.update(train_list=str(args.train_list), sample_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
    print(result)


if __name__ == "__main__":
    main()
