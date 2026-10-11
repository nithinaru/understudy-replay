// RSI: the failure labeler that improves itself (data/rsi.json, written by scripts/rsi_run.py).
// MuJoCo and the 21 gates print a line of text for every clip and rollout; an agent names each failure with one term,
// and a Bonsai loop rewrites the agent's rules, keeping a change only if held-out accuracy rises.
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => `${Math.round(x * 100)}%`;
  const term = (t) => `<span class="rsi-term t-${esc(t)}">${esc(String(t).replace(/_/g, " "))}</span>`;

  const TARGET = {
    vla: "Tuning a VLA: Runway films the demonstrations, MuJoCo and the gates write what happened, the labeler names each failure.",
    llm: "Tuning an LLM: the same loop reads the agent's own transcripts and test output instead. No Runway step; Bonsai grows the harness.",
  };

  async function boot() {
    const root = $("#rsi");
    if (!root) return;
    let d;
    try { const r = await fetch(`data/rsi.json?t=${Date.now()}`); if (!r.ok) return; d = await r.json(); } catch { return; }
    root.hidden = false;
    window.RSI = d;

    $("#rsi-score").innerHTML = `held-out <b>${pct(d.held_out.seed)}</b> → <b class="up">${pct(d.held_out.final)}</b>`;
    $("#rsi-sub").textContent = `${d.corpus.lines} lines from completed runs: ${d.corpus.gate_lines} gate reasons, ${d.corpus.mujoco_lines} MuJoCo rollouts. ${d.corpus.practice} practice, ${d.corpus.held_out} held out.`;

    // the tree, generation by generation: kept versions stay, pruned ones are struck through
    const byGen = {};
    for (const v of d.versions) (byGen[v.generation] ||= []).push(v);
    $("#rsi-tree").innerHTML = Object.keys(byGen).map((g) => `<li><span class="rsi-g">g${g}</span><div>${byGen[g].map((v) =>
      `<p class="rsi-v ${v.kept ? "kept" : "pruned"}"><b>${esc(v.id)}</b> <span class="rsi-pc">${pct(v.score)}</span> ${esc(v.change)}${
        v.shed && v.shed !== "nothing (under budget)" ? ` <span class="rsi-shed">shed ${esc(v.shed)}</span>` : ""}${
        v.signal ? `<span class="rsi-sig">${esc(v.signal)}</span>` : ""}</p>`).join("")}</div></li>`).join("");

    $("#rsi-rules").innerHTML = d.final.rules.map((r, i) => `<li><span class="rsi-n">${i + 1}</span><code>"${esc(r.phrase)}"</code> ${term(r.term)}</li>`).join("")
      + `<li><span class="rsi-n">·</span><code>otherwise</code> ${term(d.final.default)}</li>`;

    $("#rsi-fixed").innerHTML = d.fixed.slice(0, 8).map((x) => `<li><p class="rsi-line">${esc(x.text)}</p>
      <p class="rsi-was">${term(x.seed)} <span>→</span> ${term(x.final)} ${x.final === x.gold ? '<span class="ok">right</span>' : '<span class="no">still wrong</span>'}</p></li>`).join("");

    $("#rsi-runs").innerHTML = d.runs.map((r) => `<li class="rsi-run">
      <div class="rsi-rh"><b>${esc(r.run)}</b><span>${r.lines} lines · labeled right ${pct(r.seed_acc)} → ${pct(r.final_acc)}</span></div>
      <p class="rsi-trend">${esc(r.trend)}</p>
      <div class="rsi-terms">${Object.entries(r.terms).map(([t, c]) => `${term(t)}<small>${c}</small>`).join(" ")}</div>
      ${r.actions.length ? `<ul class="rsi-act">${r.actions.map((a) => `<li>${esc(a.do)} <small>(${a.count})</small></li>`).join("")}</ul>` : ""}
    </li>`).join("");
    $("#rsi-note").textContent = d.note;

    const seg = $("#rsi-target");
    const set = (k) => { for (const b of seg.children) b.setAttribute("aria-checked", String(b.dataset.k === k)); $("#rsi-target-hint").textContent = TARGET[k]; };
    seg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) set(b.dataset.k); });
    set("vla");
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", boot) : boot();
})();
