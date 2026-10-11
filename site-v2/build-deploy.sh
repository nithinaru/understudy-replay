#!/usr/bin/env bash
# Builds ../deploy-v2: the static site (no backend) with the media it replays. Re-run after the router bench updates.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
out="$repo/deploy-v2"
slug="put-the-red-block-in-the-bowl"
run="$repo/data/web-runs/$slug"

mkdir -p "$out/runs/$slug/media/vla" "$out/data" "$out/media"
cp "$here/index.html" "$here/site.css" "$here/site.js" "$here/preview.js" "$here/route.js" "$here/rsi.js" "$out/"
# the SO-101 model for the in-browser MuJoCo preview (decimated meshes, ~1.2 MB)
[ -f "$out/sim/so101/so101_new_calib.xml" ] || "$repo/.venv/bin/python" "$repo/scripts/web_meshes.py" "$out/sim/so101"
rm -rf "$out/api" && cp -R "$here/api" "$out/api"   # live Runway generation (Vercel functions; key in the RUNWAY_API_KEY env var)
cp "$here/media/hero-cine.jpg" "$out/media/"
cp "$here/data/recorded.json" "$here/data/overview.json" "$here/data/rsi.json" "$out/data/"
cp "$repo/data/router-dryruns.json" "$out/data/router-dryruns.json"
for c in v01 v03 v06 v07; do cp "$run/media/vla/$c.mp4" "$run/media/vla/$c.json" "$out/runs/$slug/media/vla/"; done
[ -f "$run/router_bench.json" ] && cp "$run/router_bench.json" "$out/runs/$slug/router_bench.json"

# the three-prompt manifest and each prompt's SmolVLA films (tiles_dir relative to its run, or absolute under it)
if [ -f "$repo/data/three-prompts.json" ]; then
  cp "$repo/data/three-prompts.json" "$out/data/three-prompts.json"
  python3 - "$repo/data/web-runs" "$out/runs" "$repo/data/three-prompts.json" <<'EOF'
import json, re, shutil, sys
from pathlib import Path
runs, dst, man = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
for e in (man if isinstance(man, list) else man.get("prompts", [])):
    slug = e.get("slug") or ""
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        continue
    d = re.sub(r"^.*/web-runs/[^/]+/", "", str(e.get("tiles_dir") or "media/vla")).strip("/")
    src = runs / slug / d
    if not src.is_dir():
        continue
    (dst / slug / d).mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        if f.is_file() and f.suffix in (".mp4", ".json"):
            shutil.copy(f, dst / slug / d / f.name)
EOF
fi

# every Runway clip per task (web-compressed) + data/gallery.json for the overview gallery
python3 "$repo/scripts/gallery_build.py"

cat > "$out/vercel.json" <<'EOF'
{ "cleanUrls": false, "headers": [{ "source": "/runs/(.*)", "headers": [{ "key": "Cache-Control", "value": "public, max-age=3600" }] }] }
EOF
du -sh "$out"
