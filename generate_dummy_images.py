#!/usr/bin/env python3
"""Generate dummy image files for smoke-testing image pipelines."""

from __future__ import annotations

import argparse
import random
import struct
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random RGB PNG images.")
    parser.add_argument("-n", "--num-images", type=int, default=100)
    parser.add_argument("-o", "--output-dir", default=".")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, rng: random.Random) -> None:
    rows = []
    base = [rng.randrange(256), rng.randrange(256), rng.randrange(256)]
    accent = [rng.randrange(256), rng.randrange(256), rng.randrange(256)]

    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            block = ((x // 32) + (y // 32)) % 2
            noise = rng.randrange(36)
            color = accent if block else base
            row.extend((channel + noise) % 256 for channel in color)
        rows.append(bytes(row))

    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", ihdr)
    data += png_chunk(b"IDAT", zlib.compress(raw, level=6))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, args.num_images + 1):
        path = output_dir / f"dummy_{index:04d}.png"
        write_png(path, args.width, args.height, rng)
        print(f"saved {path}")

    print(f"generated {args.num_images} dummy images in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
