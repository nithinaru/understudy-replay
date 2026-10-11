"""Grow the RSI failure labeler on the text of every completed run and write data/rsi.json (+ the site's copy).

  python3 scripts/rsi_run.py                       # deterministic proposer, no network
  python3 scripts/rsi_run.py --proposer claude     # Claude writes each change (needs the claude CLI)

Set RAINDROP_WRITE_KEY to trace every proposal to Raindrop. See src/understudy/rsi.py.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from understudy import rsi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--generations", type=int, default=6)
ap.add_argument("--tries", type=int, default=2)
ap.add_argument("--proposer", choices=["sim", "claude"], default="sim")
a = ap.parse_args()

out = rsi.report(rsi.grow(a.generations, a.tries, a.proposer))
for p in (ROOT / "data" / "rsi.json", ROOT / "site-v2" / "data" / "rsi.json"):
    p.write_text(json.dumps(out, indent=1))
print(f"held-out {out['held_out']['seed']:.1%} -> {out['held_out']['final']:.1%}, {len(out['versions'])} versions; wrote data/rsi.json")
