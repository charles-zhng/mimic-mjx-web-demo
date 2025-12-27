#!/usr/bin/env python3
"""Export motion reference clips to web-friendly format.

This script exports motion reference clips from HDF5 to a compact JSON/binary
format that can be loaded in the browser.

Usage:
    python export_motions.py \
        --input /path/to/rodent_reference_clips.h5 \
        --output /path/to/clips.json \
        --num-clips 10
"""

import argparse
import json
import gzip
from pathlib import Path

import h5py
import numpy as np


def export_clips(input_path: str, output_path: str, num_clips: int = 10,
                clip_length: int = 250, compress: bool = True):
    """Export motion clips to web-friendly format.

    Args:
        input_path: Path to HDF5 reference clips file
        output_path: Output JSON file path
        num_clips: Number of clips to export
        clip_length: Frames per clip
        compress: Whether to gzip the output
    """
    print(f"Loading clips from: {input_path}")

    with h5py.File(input_path, 'r') as f:
        qpos = f['qpos'][:]
        qvel = f['qvel'][:]
        xpos = f['xpos'][:]  # Body positions (num_frames, num_bodies, 3)
        xquat = f['xquat'][:]
        # Get body names for reference
        if 'names_xpos' in f:
            body_names = f['names_xpos'][()].astype(str).tolist()
            print(f"Body names: {body_names[:5]}... ({len(body_names)} total)")

    total_frames = qpos.shape[0]
    total_clips = total_frames // clip_length
    num_clips = min(num_clips, total_clips)

    print(f"Total frames: {total_frames}")
    print(f"Total clips available: {total_clips}")
    print(f"Exporting {num_clips} clips")

    # Select clips evenly distributed across the dataset
    clip_indices = np.linspace(0, total_clips - 1, num_clips, dtype=int)

    clips = []
    for i, clip_idx in enumerate(clip_indices):
        start_frame = clip_idx * clip_length
        end_frame = start_frame + clip_length

        clip_data = {
            "id": int(clip_idx),
            "name": f"clip_{clip_idx:04d}",
            "num_frames": clip_length,
            # Store as nested lists for JSON serialization
            # Use float16 precision to reduce size, then convert to list
            "qpos": np.round(qpos[start_frame:end_frame], 4).astype(np.float32).tolist(),
            "qvel": np.round(qvel[start_frame:end_frame], 4).astype(np.float32).tolist(),
            "xpos": np.round(xpos[start_frame:end_frame], 4).astype(np.float32).tolist(),
        }

        clips.append(clip_data)
        print(f"  Exported clip {clip_idx} ({i + 1}/{num_clips})")

    # Create output structure
    output_data = {
        "metadata": {
            "num_clips": num_clips,
            "clip_length": clip_length,
            "qpos_dim": int(qpos.shape[1]),
            "qvel_dim": int(qvel.shape[1]),
            "xpos_shape": list(xpos.shape[1:]),  # (num_bodies, 3)
            "mocap_hz": 50,  # From config
        },
        "clips": clips,
    }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        output_path_gz = output_path.with_suffix(".json.gz")
        print(f"Saving compressed clips to: {output_path_gz}")
        with gzip.open(output_path_gz, 'wt', encoding='utf-8') as f:
            json.dump(output_data, f, separators=(',', ':'))

        # Calculate compression stats
        uncompressed_size = len(json.dumps(output_data))
        compressed_size = output_path_gz.stat().st_size
        print(f"  Uncompressed: {uncompressed_size / 1024 / 1024:.2f} MB")
        print(f"  Compressed: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  Ratio: {compressed_size / uncompressed_size:.2%}")
    else:
        print(f"Saving clips to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, separators=(',', ':'))

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Export motion clips to JSON")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to HDF5 reference clips file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=10,
        help="Number of clips to export (default: 10)",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=250,
        help="Frames per clip (default: 250)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression",
    )

    args = parser.parse_args()

    export_clips(
        args.input,
        args.output,
        num_clips=args.num_clips,
        clip_length=args.clip_length,
        compress=not args.no_compress,
    )


if __name__ == "__main__":
    main()
