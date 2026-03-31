"""
extract_stats.py
================
Run this ONCE on your laptop (where Final_Best_model.pt lives).
It reads the TargetTransform statistics saved during training and
prints the JSON block you need to paste into app.py.

Usage:
    python extract_stats.py

Then copy everything between the ===PASTE START=== and ===PASTE END===
markers and paste it as the value of TRANSFORM_STATS in app.py.
"""

import json
import sys
from pathlib import Path

PT_PATH = Path("Saved_model/Final_Best_model.pt")

# Allow overriding path via command line
if len(sys.argv) > 1:
    PT_PATH = Path(sys.argv[1])

if not PT_PATH.exists():
    # Try Desktop path
    home = Path.home()
    PT_PATH = home / "Desktop" / "MODEL" / "Final_Best_Model" / "Final_Best_model.pt"

if not PT_PATH.exists():
    print(f"ERROR: Could not find Final_Best_model.pt")
    print(f"Tried: {PT_PATH}")
    print(f"\nUsage: python extract_stats.py <path/to/Final_Best_model.pt>")
    sys.exit(1)

print(f"Loading: {PT_PATH}")

try:
    import torch
except ImportError:
    print("ERROR: torch is not installed.")
    print("Install it with:  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

try:
    # PyTorch 2.6+ defaults to weights_only=True, which can fail for full checkpoints.
    # This script is for trusted local checkpoints, so explicitly allow full deserialization.
    ckpt = torch.load(str(PT_PATH), map_location="cpu", weights_only=False)
except TypeError:
    # Backward compatibility with older torch versions that do not expose weights_only.
    ckpt = torch.load(str(PT_PATH), map_location="cpu")

if "target_transform" not in ckpt:
    print("ERROR: 'target_transform' key not found in checkpoint.")
    print("Keys present:", list(ckpt.keys()))
    sys.exit(1)

stats = ckpt["target_transform"]

# Validate expected keys
required = {"targets", "is_count", "p99", "mean_", "std_"}
missing  = required - set(stats.keys())
if missing:
    print(f"WARNING: Missing keys in transform stats: {missing}")

# Convert to JSON-serialisable Python (torch tensors → float)
def to_python(obj):
    if hasattr(obj, "item"):   return obj.item()
    if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_python(v) for v in obj]
    return obj

clean_stats = to_python(stats)

print(f"\nFound {len(clean_stats.get('targets', []))} targets:")
for t in clean_stats.get("targets", []):
    is_c = clean_stats.get("is_count", {}).get(t, False)
    mean = clean_stats.get("mean_",    {}).get(t, "?")
    std  = clean_stats.get("std_",     {}).get(t, "?")
    p99  = clean_stats.get("p99",      {}).get(t, "n/a")
    kind = "count" if is_c else "continuous"
    print(f"  {t:<28} [{kind}]  mean={mean:.4f}  std={std:.4f}" +
          (f"  p99={p99:.1f}" if is_c else ""))

json_str = json.dumps(clean_stats, indent=2)

print("\n" + "="*60)
print("PASTE THE BLOCK BELOW INTO app.py as TRANSFORM_STATS = ...")
print("="*60)
print("\n===PASTE START===")
print(json_str)
print("===PASTE END===\n")

# Also write to a file for convenience
out_path = Path("transform_stats.json")
out_path.write_text(json_str)
print(f"Also saved to: {out_path.absolute()}")
print("\nDone. Now in app.py replace:")
print("    TRANSFORM_STATS = None")
print("with:")
print("    TRANSFORM_STATS = <the JSON above>")