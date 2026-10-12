"""Reproducible forward/training smoke demo on synthetic horizontal/vertical bars."""

import argparse
import json
import math

import torch
from torch.nn import functional as F

from .model import HybridModel
from .quantum import MAX_QUBITS


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _learning_rate(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def make_demo_batch(
    batch_size: int, image_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create alternating vertical (0) and horizontal (1) bars with light noise."""
    generator = torch.Generator().manual_seed(seed)
    images = 0.1 * torch.rand(batch_size, 1, image_size, image_size, generator=generator)
    labels = torch.arange(batch_size) % 2
    width = max(1, image_size // 6)
    for sample, label in enumerate(labels):
        start = int(torch.randint(0, image_size - width + 1, (1,), generator=generator).item())
        if label.item() == 0:
            images[sample, 0, :, start : start + width] += 0.9
        else:
            images[sample, 0, start : start + width, :] += 0.9
    return images.clamp(0, 1), labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive, default=4)
    parser.add_argument("--image-size", type=_positive, default=28)
    parser.add_argument("--qubits", type=_positive, choices=range(1, MAX_QUBITS + 1), default=4)
    parser.add_argument(
        "--steps", type=_nonnegative, default=0, help="optimizer steps; 0 = inference"
    )
    parser.add_argument("--shots", type=_positive, help="optional finite-shot Aer simulation")
    parser.add_argument("--seed", type=_nonnegative, default=7)
    parser.add_argument("--learning-rate", type=_learning_rate, default=0.01)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args(argv)
    if args.image_size < 4:
        parser.error("--image-size must be >= 4")
    if args.seed > 2**32 - 1:
        parser.error("--seed must be <= 4294967295")

    torch.manual_seed(args.seed)
    model = HybridModel(n_qubits=args.qubits, num_classes=2, shots=args.shots, seed=args.seed)
    images, labels = make_demo_batch(args.batch_size, args.image_size, args.seed)
    with torch.no_grad():
        initial_loss = F.cross_entropy(model(images), labels).item()
    losses = []
    if args.steps:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        logits = model(images)
        final_loss = F.cross_entropy(logits, labels).item()
    summary = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "n_qubits": args.qubits,
        "shots": args.shots,
        "steps": args.steps,
        "initial_loss": initial_loss,
        "training_losses": losses,
        "final_loss": final_loss,
        "labels": labels.tolist(),
        "predictions": logits.argmax(dim=1).tolist(),
        "logits": logits.tolist(),
    }
    if args.json:
        print(json.dumps(summary, allow_nan=False))
    else:
        mode = "exact statevector" if args.shots is None else f"{args.shots} shots"
        print(f"{args.batch_size} images | {args.qubits} qubits | {mode} | seed {args.seed}")
        print(f"Loss: {initial_loss:.6f} -> {final_loss:.6f} ({args.steps} training steps)")
        print(f"Labels:      {summary['labels']}")
        print(f"Predictions: {summary['predictions']}")
        print(f"Logits:\n{logits}")
