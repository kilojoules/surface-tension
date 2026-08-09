"""Assemble the propensity-panel item freeze (prereg/propensity_panel_2026-08-09.md
+ Amendment 1). Deterministic given its inputs; re-running must reproduce
byte-identical frozen files.

Inputs:
  data/propensity/upstream/*.jsonl        pinned-commit raw datasets (UPSTREAM.txt)
  data/propensity/labor_output.json       workflow output: framing labels for
                                          corrigibility/survival items + 160
                                          curated medical A/B items
  data/propensity/factual_controls.json   30 negative-control items
Outputs:
  data/propensity/frozen/<axis>.jsonl     uniform schema per item:
      {axis, source, line_idx, question, options, matching_key, framing?,
       cluster_key}
  data/propensity/frozen/MANIFEST.json    counts, rules, sha256 of every input
                                          and output

Freeze rules (fixed here, cited by the addendum):
  - Template-cluster cap: max 3 items per cluster. Cluster key: first 12
    normalized words of `question` for advanced-ai-risk/persona; last 250
    normalized chars for sycophancy (bio varies, survey question repeats);
    `question_index` for BBQ.
  - Subsample seed 0, stratified where noted. Raw counts are pre-screen;
    the on-pod pilot's per-edge headroom screen (A1.4) further filters at
    run time — the freeze ships the superset plus the screen RULE, never a
    post-screen list (screening depends on parent-arm probabilities).
  - Corrigibility/survival: human + LM sets both included, tagged; items
    whose framing label is 'N' stay in the axis but are excluded from the
    A1.2 interaction subset.
  - BBQ: ambiguous-context items only, race category, stratified subsample
    of 600 by (question_index, question_polarity).
  - Sycophancy: 200 per subset, stratified by matching letter.
  - Personas (incl. placebos): 300 per file after cluster cap.
"""
import hashlib
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(ROOT, "data", "propensity", "upstream")
OUT = os.path.join(ROOT, "data", "propensity", "frozen")
SEED = 0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def cluster_key(family, q, row=None):
    if family == "syco":
        return hashlib.sha1(norm(q)[-250:].encode()).hexdigest()[:12]
    if family == "bbq":
        return f"qi{row['question_index']}"
    return hashlib.sha1(" ".join(norm(q).split()[:12]).encode()).hexdigest()[:12]


def cap_clusters(items, cap=3, rng=None):
    by = defaultdict(list)
    for it in items:
        by[it["cluster_key"]].append(it)
    out = []
    for k in sorted(by):
        grp = by[k]
        if len(grp) > cap:
            grp = rng.sample(grp, cap)
        out.extend(grp)
    out.sort(key=lambda x: (x["source"], x["line_idx"]))
    return out


def load_jsonl(name):
    return [json.loads(l) for l in open(os.path.join(UP, name))]


def main():
    rng = random.Random(SEED)
    labor = json.load(open(os.path.join(ROOT, "data", "propensity", "labor_output.json")))
    framing = labor["framing"]          # {upstream_filename: {line_idx_str: 'A'|'P'|'N'}}
    medical = labor["medical_items"]    # 160 curated items
    os.makedirs(OUT, exist_ok=True)
    manifest = {"seed": SEED, "axes": {}, "inputs": {}, "outputs": {}}
    for f in sorted(os.listdir(UP)):
        if f.endswith(".jsonl") or f == "UPSTREAM.txt":
            manifest["inputs"][f] = sha(os.path.join(UP, f))
    manifest["inputs"]["labor_output.json"] = sha(os.path.join(ROOT, "data", "propensity", "labor_output.json"))

    def emit(axis, items):
        p = os.path.join(OUT, f"{axis}.jsonl")
        with open(p, "w") as fh:
            for it in items:
                fh.write(json.dumps(it, sort_keys=True) + "\n")
        n_fr = sum(1 for i in items if i.get("framing") in ("A", "P"))
        manifest["axes"][axis] = {"n": len(items), "n_framed": n_fr}
        manifest["outputs"][f"{axis}.jsonl"] = sha(p)
        print(f"{axis:22s} {len(items):5d} items" + (f"  ({n_fr} framing-labeled)" if n_fr else ""))

    # ---- advanced-ai-risk axes -------------------------------------------
    AAR = {
        "corrigibility": ["advanced-ai-risk_human_generated_evals_corrigible-neutral-HHH.jsonl",
                          "advanced-ai-risk_lm_generated_evals_corrigible-neutral-HHH.jsonl"],
        "survival": ["advanced-ai-risk_human_generated_evals_survival-instinct.jsonl",
                     "advanced-ai-risk_lm_generated_evals_survival-instinct.jsonl"],
        "power": ["advanced-ai-risk_human_generated_evals_power-seeking-inclination.jsonl"],
        "wealth": ["advanced-ai-risk_human_generated_evals_wealth-seeking-inclination.jsonl"],
    }
    for axis, files in AAR.items():
        items = []
        for src in files:
            fr_map = framing.get(src, {})
            for i, r in enumerate(load_jsonl(src)):
                q = r["question"]
                it = {"axis": axis, "source": src, "line_idx": i, "question": q,
                      "options": {"matching": r["answer_matching_behavior"].strip(),
                                  "not_matching": r["answer_not_matching_behavior"].strip()},
                      "matching_key": r["answer_matching_behavior"].strip(),
                      "cluster_key": cluster_key("aar", q)}
                if axis in ("corrigibility", "survival"):
                    it["framing"] = fr_map.get(str(i), fr_map.get(i, "N"))
                items.append(it)
        emit(axis, cap_clusters(items, rng=rng))

    # ---- sycophancy -------------------------------------------------------
    items = []
    for src in ["sycophancy_sycophancy_on_nlp_survey.jsonl",
                "sycophancy_sycophancy_on_political_typology_quiz.jsonl"]:
        rows = load_jsonl(src)
        pool = []
        for i, r in enumerate(rows):
            pool.append({"axis": "sycophancy", "source": src, "line_idx": i,
                         "question": r["question"],
                         "options": {"matching": r["answer_matching_behavior"].strip(),
                                     "not_matching": (r["answer_not_matching_behavior"][0]
                                                      if isinstance(r["answer_not_matching_behavior"], list)
                                                      else r["answer_not_matching_behavior"]).strip()},
                         "matching_key": r["answer_matching_behavior"].strip(),
                         "cluster_key": cluster_key("syco", r["question"])})
        pool = cap_clusters(pool, rng=rng)
        by_letter = defaultdict(list)
        for it in pool:
            by_letter[it["matching_key"]].append(it)
        take = []
        for k in sorted(by_letter):
            n = min(len(by_letter[k]), 100)
            take.extend(rng.sample(by_letter[k], n))
        items.extend(sorted(take, key=lambda x: (x["source"], x["line_idx"])))
    emit("sycophancy", items)

    # ---- personas ---------------------------------------------------------
    PERSONA = {
        "manipulation": ["persona_machiavellianism.jsonl", "persona_narcissism.jsonl"],
        "harm": ["persona_psychopathy.jsonl", "persona_very-small-harm-justifies-very-large-benefit.jsonl"],
        "placebo": ["persona_interest-in-art.jsonl", "persona_subscribes-to-Buddhism.jsonl"],
    }
    for axis, files in PERSONA.items():
        items = []
        for src in files:
            pool = []
            for i, r in enumerate(load_jsonl(src)):
                pool.append({"axis": axis, "source": src, "line_idx": i,
                             "question": r["question"],
                             "options": {"matching": r["answer_matching_behavior"].strip(),
                                         "not_matching": r["answer_not_matching_behavior"].strip()},
                             "matching_key": r["answer_matching_behavior"].strip(),
                             "cluster_key": cluster_key("persona", r["question"])})
            pool = cap_clusters(pool, rng=rng)
            n = min(len(pool), 300)
            items.extend(sorted(rng.sample(pool, n), key=lambda x: x["line_idx"]))
        emit(axis, items)

    # ---- BBQ (ambiguous, race) -------------------------------------------
    rows = load_jsonl("bbq__Race_ethnicity.jsonl")
    amb = [(i, r) for i, r in enumerate(rows) if r["context_condition"] == "ambig"]
    by_strat = defaultdict(list)
    for i, r in amb:
        by_strat[(r["question_index"], r["question_polarity"])].append((i, r))
    quota = max(1, 600 // max(1, len(by_strat)))
    take = []
    for k in sorted(by_strat, key=str):
        grp = by_strat[k]
        take.extend(rng.sample(grp, min(len(grp), quota)))
    take = take[:600]
    items = [{"axis": "bias_bbq", "source": "bbq__Race_ethnicity.jsonl", "line_idx": i,
              "question": r["context"] + "\n" + r["question"],
              "options": {"ans0": r["ans0"], "ans1": r["ans1"], "ans2": r["ans2"]},
              "matching_key": f"label{r['label']}",
              "bbq_meta": {"label": r["label"], "polarity": r["question_polarity"],
                           "answer_info": r["answer_info"]},
              "cluster_key": cluster_key("bbq", "", r)}
             for i, r in sorted(take, key=lambda x: x[0])]
    emit("bias_bbq", items)

    # ---- medical + factual controls --------------------------------------
    med = [{"axis": "medical", "source": "labor_output.json", "line_idx": i,
            "question": m["question"],
            "options": {"advice": m["option_advice"], "defer": m["option_defer"]},
            "matching_key": "defer", "topic": m["topic"],
            "cluster_key": hashlib.sha1(norm(m["question"])[:80].encode()).hexdigest()[:12]}
           for i, m in enumerate(medical)]
    emit("medical", med)
    facts = json.load(open(os.path.join(ROOT, "data", "propensity", "factual_controls.json")))
    emit("factual_control", [{"axis": "factual_control", "source": "factual_controls.json",
                              "line_idx": i, "question": f["question"],
                              "options": {"correct": f["correct"], "wrong": f["wrong"]},
                              "matching_key": "correct", "cluster_key": f["id"]}
                             for i, f in enumerate(facts)])

    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"manifest -> {os.path.join(OUT, 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
