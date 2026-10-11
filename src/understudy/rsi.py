"""RSI: a failure labeler that rewrites its own rules, Bonsai style, on the text MuJoCo and the gates print.

The last stage of Understudy turns physics into text: every gate that rejects a Runway clip writes a reason, and
every MuJoCo rollout of a trained SmolVLA ends in a log line. An agent reads each line and names the failure with
one term from a fixed taxonomy. Those terms are what the data step acts on (fix the Runway prompt, re-anchor a clip,
add episodes at the starts the policy misses).

The agent's harness is a small JSON file: ordered rules (a phrase and the term it means, first match wins), a
default term and a rule budget. It improves itself with the Bonsai loop (github.com/... bonsai, loop.py):
  1. run the best harness on the practice lines
  2. a proposer reads the mislabeled lines and writes ONE change; whatever it adds, it sheds a rule of similar size
     when the budget is full, so the harness never grows past it
  3. score the child on held-out lines it has never seen
  4. keep it only if held-out accuracy rises; otherwise prune it, and the cut stays in the record

The gold term for each line comes from the structured fields behind it (gate id, lifted, success), never from the
text, so the grade is right or wrong. The proposer is either a deterministic phrase miner ("sim", no network) or
Claude through the `claude` CLI ("claude"). With RAINDROP_WRITE_KEY set, every proposal is traced to Raindrop and
marked kept or pruned.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TERMS = {
    "success": "the cube ends in the bowl (bowl inner radius 5 cm: resting under 5 cm from its centre is in)",
    "duplicated_object": "the video adds or splits a red block",
    "tracking_drift": "the block moves without the hand",
    "grasp_miss": "the cube never leaves the table",
    "missed_bowl": "lifted, but it rests outside the bowl (5 cm or more from its centre)",
    "reach_limit": "the SO-101 cannot reach the pose",
    "motion_too_fast": "joint speed, acceleration or jerk over the limit",
}

# gate id -> term. The gates are in src/understudy/gates.py; their limits are in docs/DATA-SPEC.md.
GATE_TERM = {
    "scene": "duplicated_object", "one_object": "duplicated_object", "no_duplicate": "duplicated_object",
    "video_in_bowl": "duplicated_object", "coupled": "tracking_drift", "carry": "grasp_miss", "lift": "grasp_miss",
    "events": "grasp_miss", "close_before_lift": "grasp_miss", "ik": "reach_limit", "vel": "motion_too_fast",
    "acc": "motion_too_fast", "jerk": "motion_too_fast", "open_over_bowl": "missed_bowl", "sim_success": "missed_bowl",
}  # fmt: skip

# What the data step does about each term. Nothing here is executed: the report lists it per run.
ACTION = {
    "duplicated_object": "add 'exactly one red block; it never splits or duplicates' to the Runway prompt (PhyT2V)",
    "tracking_drift": "drop the clip: the tracked block is not the one the hand holds",
    "grasp_miss": "add re-anchored episodes at the cube starts where the grasp missed",
    "missed_bowl": "add episodes with a longer carry and a slower release over the bowl",
    "reach_limit": "re-anchor the clip closer to the SO-101 workspace",
    "motion_too_fast": "time-stretch the retarget to bring joint speed under the limit",
}

# Prior runs, all completed end to end. The gate reasons come from the Runway clip runs; the rollouts from MuJoCo
# evaluations of each checkpoint on 50 unseen cube starts.
EVALS = {
    "smolvla-base (zero-shot)": "data/smolvla/base-zeroshot/eval.json",
    "smolvla-base (out of the box)": "data/smolvla/base-ootb/eval.json",
    "understudy-650 (recorded run)": "data/smolvla/understudy-650-record/eval.json",
    "expert-3000": "data/probes/bench/expert-3000.eval.json",
    "expert-20000": "data/probes/bench/expert-20000.eval.json",
}
GATE_RUNS = {"runway clips: put the red block in the bowl": "data/runs/put-the-red-block-in-the-bowl/results"}

TARGET_XY, FRAMES_CAP = (0.25, 0.0), 750

# Rollout log phrasings. The policy log varies its wording, so the labeler cannot rely on one fixed sentence.
SAY = {
    "success": [
        "seed {s}: cube lifted and placed, rests {d:.1f} cm from bowl centre after {f} frames",
        "seed {s}: episode done at frame {f}, cube settled inside the bowl",
        "seed {s}: placed in {t:.1f} s, cube {d:.1f} cm off centre, gripper open",
    ],
    "grasp_miss": [
        "seed {s}: cube never left the table, {f} frames",
        "seed {s}: gripper closed on air; cube stayed at z {z:.3f} m after {f} frames",
        "seed {s}: no lift detected, time limit hit at frame {f}",
    ],
    "missed_bowl": [
        "seed {s}: cube lifted but rests {d:.1f} cm from bowl centre",
        "seed {s}: released early, cube landed outside the bowl rim ({d:.1f} cm)",
        "seed {s}: cube carried, dropped {d:.1f} cm short, time limit hit at frame {f}",
    ],
}

SEED = {
    "rules": [
        {"phrase": "duplicat", "term": "duplicated_object"},
        {"phrase": "joint", "term": "motion_too_fast"},
        {"phrase": "bowl", "term": "success"},
    ],
    "default": "grasp_miss",
    "max_rules": 8,
}


def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def corpus(root: Path = ROOT) -> list[dict]:
    """Every text line the last stage produced across the prior runs, with its gold term."""
    lines = []
    for run, rel in GATE_RUNS.items():
        for f in sorted((root / rel).glob("*.json")):
            r = json.loads(f.read_text())
            for g in r["gates"]:
                if not g["passed"] and g["id"] in GATE_TERM:
                    lines.append({"run": run, "id": f"{r['clip_id']}:{g['id']}", "text": f"{r['clip_id']} {g['why']}",
                                  "gold": GATE_TERM[g["id"]], "source": "gate"})  # fmt: skip
            if r["accepted"]:
                lines.append({"run": run, "id": f"{r['clip_id']}:all", "text": f"{r['clip_id']} passed all 21 gates; cube ends in the bowl in MuJoCo",
                              "gold": "success", "source": "gate"})  # fmt: skip
    for run, rel in EVALS.items():
        p = root / rel
        if not p.exists():
            continue
        for e in json.loads(p.read_text())["episodes"]:
            gold = "success" if e["success"] else "missed_bowl" if e["lifted"] else "grasp_miss"
            x, y, z = e["cube_end"]
            d = 100 * math.hypot(x - TARGET_XY[0], y - TARGET_XY[1])
            k = _h(f"{run}{e['seed']}")
            text = SAY[gold][k % 3].format(s=e["seed"], d=d, f=e["frames"], t=e["frames"] / 30, z=z)
            lines.append({"run": run, "id": f"{run}:{e['seed']}", "text": text, "gold": gold, "source": "mujoco"})
    return lines


def split(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """Practice and held-out halves, fixed by a hash of each line's id."""
    practice = [x for x in lines if _h(x["id"]) % 2 == 0]
    return practice, [x for x in lines if _h(x["id"]) % 2 == 1]


def label(h: dict, text: str) -> tuple[str, int | None]:
    t = text.lower()
    for i, r in enumerate(h["rules"]):
        if r["phrase"].lower() in t:
            return r["term"], i
    return h["default"], None


def score(h: dict, lines: list[dict]) -> dict:
    rows = []
    for x in lines:
        got, rule = label(h, x["text"])
        rows.append({**x, "got": got, "rule": rule, "ok": got == x["gold"]})
    return {"acc": sum(r["ok"] for r in rows) / max(len(rows), 1), "rows": rows}


# ---------- proposers ----------
def _phrases(text: str) -> set[str]:
    w = re.findall(r"[a-z][a-z'-]+", text.lower())
    return {" ".join(w[i : i + n]) for n in (1, 2, 3) for i in range(len(w) - n + 1)}


def propose_sim(h: dict, res: dict, pruned: list[str], i: int) -> dict:
    """Deterministic proposer: take the most common confusion on practice, mine the phrase that best separates its
    lines from every other term, and put a rule for it first. Try i takes the i-th best phrase."""
    fails = [r for r in res["rows"] if not r["ok"]]
    if not fails:
        raise RuntimeError("nothing to fix on practice")
    have = {r["phrase"] for r in h["rules"]}
    cands = []
    for (gold, got), _ in Counter((r["gold"], r["got"]) for r in fails).most_common(3):
        target = [r for r in fails if r["gold"] == gold and r["got"] == got]
        others = [r for r in res["rows"] if r["gold"] != gold]
        hits = Counter(p for r in target for p in _phrases(r["text"]))
        bad = Counter(p for r in others for p in _phrases(r["text"]))
        for p, c in hits.items():
            if p not in have and len(p) > 3 and f"'{p}' -> {gold}" not in " ".join(pruned):
                cands.append((c - 3 * bad[p], -len(p), p, gold, got, len(target)))
    if not cands:
        raise RuntimeError("no phrase separates the mistakes")
    _, _, phrase, gold, got, n = sorted(cands, reverse=True)[min(i, len(cands) - 1)]
    target = range(n)
    rules = [{"phrase": phrase, "term": gold}, *h["rules"]]
    shed = "nothing (under budget)"
    if len(rules) > h["max_rules"]:
        used = Counter(r["rule"] for r in res["rows"] if r["rule"] is not None)
        # cut the rule that earns least on practice (fewest correct labels), never the new one
        earn = [(sum(1 for r in res["rows"] if r["rule"] == j and r["ok"]), used[j], j) for j in range(len(h["rules"]))]
        _, _, j = min(earn)
        shed = f"rule '{h['rules'][j]['phrase']}' -> {h['rules'][j]['term']}"
        rules.pop(j + 1)
    return {
        "change": f"add '{phrase}' -> {gold} first",
        "shed": shed,
        "signal": f"{len(target)} {gold} lines labeled {got}",
        "harness": {**h, "rules": rules},
    }


PROMPT = """You improve a text labeler by editing its harness. The harness is JSON:
{{"rules": ordered [{{"phrase": substring, "term": one of {terms}}}], first match wins, case-insensitive;
"default": term when nothing matches; "max_rules": {cap} (fixed)}}

Current harness:
{harness}

On {n} practice lines it mislabeled {k}:
{fails}

Changes already tried and pruned (do not repeat): {pruned}

Make exactly ONE change aimed at the most common mistake. If the rules would pass max_rules, also remove one rule
that is not earning its place. Do not quote seed numbers or clip ids: the change is scored on lines you have not seen.
Reply with JSON only: {{"change": "one sentence", "shed": "what was removed", "signal": "the failure pattern", "harness": {{...}}}}"""


def propose_claude(h: dict, res: dict, pruned: list[str], i: int, model: str = "claude-opus-5-5") -> dict:
    fails = [r for r in res["rows"] if not r["ok"]][:40]
    prompt = PROMPT.format(
        terms=list(TERMS), cap=h["max_rules"], harness=json.dumps(h, indent=1), n=len(res["rows"]), k=len(fails),
        fails="\n".join(f"- {r['text']!r}: labeled {r['got']}, should be {r['gold']}" for r in fails),
        pruned="; ".join(pruned) or "(none)",
    )  # fmt: skip
    claude = shutil.which("claude") or "/usr/local/bin/claude"
    out = subprocess.run([claude, "-p", prompt, "--model", model, "--output-format", "json", "--max-turns", "1"],
                         capture_output=True, text=True, timeout=300)  # fmt: skip
    text = json.loads(out.stdout)["result"]
    reply = json.loads(text[text.index("{") : text.rindex("}") + 1])
    child = reply["harness"]
    child["rules"] = [r for r in child["rules"] if r.get("term") in TERMS][: h["max_rules"]]
    child["max_rules"] = h["max_rules"]
    if child.get("default") not in TERMS:
        child["default"] = h["default"]
    reply["harness"] = child
    reply["prompt"] = prompt
    reply["output"] = text
    return reply


def _raindrop():
    if not os.getenv("RAINDROP_WRITE_KEY"):
        return None
    try:
        import raindrop.analytics as rd

        rd.init(os.environ["RAINDROP_WRITE_KEY"])
        return rd
    except Exception:
        return None


# ---------- the loop ----------
def grow(generations: int = 6, tries: int = 2, proposer: str = "sim", root: Path = ROOT, log=print) -> dict:
    lines = corpus(root)
    practice, held = split(lines)
    rd = _raindrop()
    best = {**SEED, "id": "g0"}
    h0 = score(best, held)
    best_acc = h0["acc"]
    versions = [{"id": "g0", "parent": None, "generation": 0, "change": "seed labeler", "shed": None, "signal": None,
                 "kept": True, "score": best_acc, "rules": len(best["rules"]), "harness": SEED, "trace_id": None, "ts": time.time()}]  # fmt: skip
    log(f"g0 seed held-out {best_acc:.1%}")
    pruned, parent_of_pruned = [], "g0"
    for gen in range(1, generations + 1):
        run = score(best, practice)
        if run["acc"] == 1.0:
            log(f"g{gen}: practice is clean, stopping")
            break
        if best["id"] != parent_of_pruned:
            pruned, parent_of_pruned = [], best["id"]
        for i in range(tries):
            vid = f"g{gen}{'abcdef'[i]}"
            try:
                p = propose_claude(best, run, pruned, i) if proposer == "claude" else propose_sim(best, run, pruned, i)
            except Exception as e:  # an unusable proposal is a pruned branch, not a crashed run
                log(f"{vid} no proposal: {e}")
                break
            child = {**p["harness"], "id": vid}
            res = score(child, held)
            kept = res["acc"] > best_acc
            trace = None
            if rd:
                trace = rd.track_ai(user_id=f"labeler-{best['id']}", event="labeler_proposal", model=proposer,
                                    input=p.get("prompt", p["signal"]), output=p.get("output", p["change"]),
                                    properties={"parent": best["id"], "change": p["change"], "shed": p["shed"]})  # fmt: skip
                rd.track_signal(event_id=trace, name="task_outcome", signal_type="feedback",
                                sentiment="POSITIVE" if kept else "NEGATIVE", comment=f"{res['acc']:.0%} held-out")  # fmt: skip
            versions.append({"id": vid, "parent": best["id"], "generation": gen, "change": p["change"], "shed": p["shed"],
                             "signal": p["signal"], "kept": kept, "score": res["acc"], "rules": len(child["rules"]),
                             "harness": {k: v for k, v in child.items() if k != "id"}, "trace_id": trace, "ts": time.time()})  # fmt: skip
            log(f"{vid} {'KEPT' if kept else 'PRUNED'} {res['acc']:.1%} (best {best_acc:.1%}) {p['change']}; shed {p['shed']}")
            if kept:
                best, best_acc = child, res["acc"]
                break
            pruned.append(p["change"])
    if rd:
        rd.flush()
    final = {k: v for k, v in best.items() if k != "id"}
    return {"lines": lines, "practice": len(practice), "held": len(held), "versions": versions, "best": best["id"],
            "final": final, "seed_acc": h0["acc"], "final_acc": best_acc}  # fmt: skip


def report(g: dict) -> dict:
    """Label every line of every prior run with the seed and the final harness, and turn the labels into trends and
    data actions per run."""
    seed, final = score(SEED, g["lines"]), score(g["final"], g["lines"])
    runs = {}
    for a, b in zip(seed["rows"], final["rows"]):
        r = runs.setdefault(a["run"], {"run": a["run"], "source": a["source"], "lines": 0, "seed_ok": 0, "final_ok": 0,
                                       "terms": Counter(), "examples": {}})  # fmt: skip
        r["lines"] += 1
        r["seed_ok"] += a["ok"]
        r["final_ok"] += b["ok"]
        r["terms"][b["got"]] += 1
        r["examples"].setdefault(b["got"], b["text"])
    out = []
    for r in runs.values():
        n = r["lines"]
        fails = {t: c for t, c in r["terms"].most_common() if t != "success"}
        top = next(iter(fails), None)
        out.append({
            "run": r["run"], "source": r["source"], "lines": n,
            "seed_acc": r["seed_ok"] / n, "final_acc": r["final_ok"] / n,
            "terms": dict(r["terms"].most_common()),
            "trend": (f"{fails[top]} of {n} lines are {top}" if top else "no failures"),
            "actions": [{"term": t, "count": c, "do": ACTION[t], "example": r["examples"][t]} for t, c in fails.items() if t in ACTION],
        })  # fmt: skip
    sample = [{"text": a["text"], "gold": a["gold"], "seed": a["got"], "final": b["got"], "run": a["run"]}
              for a, b in zip(seed["rows"], final["rows"]) if a["got"] != b["got"]][:24]  # fmt: skip
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "terms": TERMS,
        "corpus": {"lines": len(g["lines"]), "practice": g["practice"], "held_out": g["held"],
                   "gate_lines": sum(x["source"] == "gate" for x in g["lines"]),
                   "mujoco_lines": sum(x["source"] == "mujoco" for x in g["lines"])},  # fmt: skip
        "held_out": {"seed": g["seed_acc"], "final": g["final_acc"]},
        "versions": g["versions"], "best": g["best"], "final": g["final"], "runs": out, "fixed": sample,
        "note": "Simulated: the labeler and its loop run for real on the text of completed runs; the data actions are proposals, nothing was retrained.",
    }
