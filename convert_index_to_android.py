"""
convert_index_to_android.py
===========================
Converts index.npz to a binary format readable by Android.

Output format (index.bin):
- 4 bytes: N (int32, big-endian) - number of pills
- 4 bytes: D (int32, big-endian) - dimension of vectors
- N * D * 4 bytes: vectors (float32, little-endian)
- N strings: ids (2-byte length prefix + UTF-8 bytes)
- N strings: colors
- N strings: logos

Usage:
  python convert_index_to_android.py --input output/index.npz --output PillIdentifier/app/src/main/assets/index.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def write_string(f, s: str):
    """Write length-prefixed UTF-8 string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('>h', len(encoded)))  # 2-byte length, big-endian
    f.write(encoded)


def convert_index(input_path: str, output_path: str):
    # Load NPZ
    data = np.load(input_path, allow_pickle=False)

    ids = data['ids']
    vectors = data['vectors'].astype(np.float32)
    colors = data['colors'] if 'colors' in data else np.array([''] * len(ids))
    logos = data['logos'] if 'logos' in data else np.array([''] * len(ids))

    n, d = vectors.shape
    print(f"[INFO] Loaded {n} pills × {d} dims")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('>i', n))  # N, big-endian int32
        f.write(struct.pack('>i', d))  # D, big-endian int32

        # Vectors (little-endian float32 for efficiency)
        for vec in vectors:
            f.write(vec.tobytes())  # numpy default is native (little-endian on most systems)

        # IDs
        for pill_id in ids:
            write_string(f, str(pill_id))

        # Colors
        for color in colors:
            write_string(f, str(color))

        # Logos
        for logo in logos:
            write_string(f, str(logo))

    output_size = Path(output_path).stat().st_size
    print(f"[OK] Written to {output_path}")
    print(f"     Size: {output_size / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert index.npz to Android binary format")
    parser.add_argument("--input", default="output/index.npz", help="Input NPZ file")
    parser.add_argument("--output", default="PillIdentifier/app/src/main/assets/index.bin",
                        help="Output binary file")
    args = parser.parse_args()

    convert_index(args.input, args.output)


if __name__ == "__main__":
    main()
