import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from understudy import rsi


def test_corpus_and_loop_improve_held_out():
    g = rsi.grow(generations=4, tries=2, log=lambda *_: None)
    assert len(g["lines"]) > 200
    assert g["final_acc"] > g["seed_acc"]
    assert len(g["final"]["rules"]) <= g["final"]["max_rules"]
    out = rsi.report(g)
    assert {r["run"] for r in out["runs"]} >= {"expert-3000", "runway clips: put the red block in the bowl"}


def test_gold_never_from_text():
    h = {"rules": [], "default": "success", "max_rules": 8}
    lines = rsi.corpus()
    assert rsi.score(h, lines)["acc"] < 1.0
