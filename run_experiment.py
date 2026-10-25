"""Main experiment runner for ICS-LLM-TM.

Produces the tables in the paper:
  Table 4  attack success rate (ASR) and OT-write rate by class, baseline vs hardened
  Table 5  leave-one-out mitigation ablation (ASR, OT-write, benign TSR)
  plus 95% Wilson CIs and McNemar paired tests (baseline vs hardened).

Usage (mock, no network -- validates the pipeline):
  python run_experiment.py --provider mock --reps 20

Usage (real model via OpenAI-compatible API):
  export OPENAI_API_KEY=...        # or AZURE/GROQ/etc.
  python run_experiment.py --provider openai --model gpt-4o-mini --reps 10
  python run_experiment.py --provider openai --model llama-3.1-8b-instruct \
         --base-url http://localhost:11434/v1 --reps 10     # local ollama/vLLM

Everything is written to results/ : raw.jsonl, table4.csv, table5.csv,
summary.json, and a run_meta.json recording the dataset source and settings.
"""
import argparse
import json
import os
import random
import statistics

from plant import Plant
from corpus import build_corpus, Retriever
from payloads import load_carriers
from attacks import build_cases, build_benign
from mcp_tools import Orchestrator, ALL_MITIGATIONS
from models import ApiModel, MockModel


def make_model(args, seed):
    if args.provider == "mock":
        return MockModel(temperature=args.temperature, seed=seed, compliance=args.mock_compliance)
    base = args.base_url
    if args.provider == "azure":
        raise SystemExit("For Azure, use --provider openai with --base-url set to your Azure endpoint/v1.")
    return ApiModel(args.model, base_url=base, api_key_env=args.api_key_env,
                    temperature=args.temperature, max_tokens=args.max_tokens)


def run_attack_case(case, model, retriever_kind, mitigations):
    plant = Plant()
    retr = Retriever(build_corpus(), kind=retriever_kind)
    orch = Orchestrator(plant, retr, model, mitigations,
                        poisoned_tools=case["poisoned"])
    res = orch.run(case["operator_msg"], injected_docs=case["injected_docs"])
    success = bool(case["success"](plant, res["final"]))
    return {"success": success, "ot_write": res["ot_write_attempted"], "final": res["final"]}


def run_benign_case(task, model, retriever_kind, mitigations):
    plant = Plant()
    retr = Retriever(build_corpus(), kind=retriever_kind)
    orch = Orchestrator(plant, retr, model, mitigations, elevated_session=False)
    res = orch.run(task["msg"], injected_docs=[])
    return {"success": bool(task["check"](plant, res["final"])), "final": res["final"]}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (100 * p, 100 * max(0, centre - half), 100 * min(1, centre + half))


def mcnemar(paired):
    """paired: list of (baseline_success, hardened_success) booleans -> (b, c, pvalue)."""
    from statsmodels.stats.contingency_tables import mcnemar as mcn
    import numpy as np
    b = sum(1 for a, h in paired if a and not h)   # fixed by hardening
    c = sum(1 for a, h in paired if not a and h)   # broken by hardening
    table = np.array([[sum(1 for a, h in paired if a and h), b],
                      [c, sum(1 for a, h in paired if not a and not h)]])
    res = mcn(table, exact=(b + c) < 25, correction=True)
    return b, c, float(res.pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock", "openai", "groq", "together", "azure"], default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--reps", type=int, default=20, help="repetitions per case")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--n-per-class", type=int, default=6)
    ap.add_argument("--retriever", choices=["tfidf", "minilm"], default="tfidf")
    ap.add_argument("--mock-compliance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--configs", default="all",
                    help="Comma list of configs to run, e.g. baseline,hardened  "
                         "(default all = baseline, hardened and the 7 leave-one-out ablations).")
    ap.add_argument("--carriers-csv", default=None,
                    help="Optional local CSV of injection carriers (columns: text[,label]).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)

    carriers, carrier_source = load_carriers(seed=args.seed,
                                             cache_path=os.path.join(args.outdir, "carriers_used.txt"),
                                             local_csv=args.carriers_csv)
    cases = build_cases(carriers, n_per_class=args.n_per_class)
    benign = build_benign(n=12)

    configs = {"baseline": [], "hardened": ALL_MITIGATIONS}
    for m in ALL_MITIGATIONS:
        configs[f"-{m}"] = [x for x in ALL_MITIGATIONS if x != m]
    if args.configs != "all":
        want = [c.strip() for c in args.configs.split(",") if c.strip()]
        bad = [c for c in want if c not in configs]
        if bad:
            raise SystemExit(f"Unknown config(s) {bad}. Choose from {list(configs)}")
        for c in ("baseline", "hardened"):
            if c not in want:
                want.insert(0, c)   # always needed for the headline comparison
        configs = {c: configs[c] for c in want}

    raw_path = os.path.join(args.outdir, "raw.jsonl")
    done_keys = set()
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_keys.add((d["config"], d["rep"], d["id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_keys:
            print(f"Resuming: {len(done_keys)} runs already in {raw_path} will be skipped.")
    raw = open(raw_path, "a", encoding="utf-8")

    # results[config][cls] = list of success bools ; ot[config][cls] = list of ot_write bools
    results = {c: {} for c in configs}
    otwrite = {c: {} for c in configs}
    paired = {c: [] for c in configs}  # for McNemar vs baseline, per rep/case (hardened + ablations)
    benign_res = {c: [] for c in configs}

    total = len(configs) * (len(cases) + len(benign)) * args.reps
    done = 0
    for cfg, mits in configs.items():
        for rep in range(args.reps):
            model = make_model(args, seed=args.seed + rep)
            for case in cases:
                done += 1
                if (cfg, rep, case["id"]) in done_keys:
                    continue
                r = run_attack_case(case, model, args.retriever, mits)
                raw.write(json.dumps({"config": cfg, "rep": rep, "id": case["id"], "cls": case["cls"],
                                      "kind": "attack", **{k: r[k] for k in ("success", "ot_write")}}) + "\n")
                raw.flush()
            for task in benign:
                done += 1
                if (cfg, rep, task["id"]) in done_keys:
                    continue
                r = run_benign_case(task, model, args.retriever, mits)
                raw.write(json.dumps({"config": cfg, "rep": rep, "id": task["id"], "kind": "benign",
                                      "success": r["success"]}) + "\n")
                raw.flush()
        print(f"  [{cfg}] done ({done}/{total} runs)")
    raw.close()

    classes = sorted({c["cls"] for c in cases})

    def flat(d, cfg):
        return [v for cls in classes for v in d[cfg].get(cls, [])]

    # Rebuild all statistics from raw.jsonl (includes runs from earlier, resumed sessions).
    rows = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("config") in configs:
                rows.setdefault((d["config"], d["rep"], d["id"]), d)
    # Sort by (rep, case id) so baseline/hardened lists are aligned for McNemar pairing.
    for (cfg, rep, cid) in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
        d = rows[(cfg, rep, cid)]
        if d["kind"] == "attack":
            results[cfg].setdefault(d["cls"], []).append(bool(d["success"]))
            otwrite[cfg].setdefault(d["cls"], []).append(bool(d["ot_write"]))
        else:
            benign_res[cfg].append(bool(d["success"]))
    incomplete = [c for c in configs if len(flat(results, c)) != len(cases) * args.reps]
    if incomplete:
        print(f"WARNING: incomplete configurations {incomplete}; rerun the same command to finish.")

    # ---- Table 4: ASR + OT-write by class, baseline vs hardened ----
    classes = sorted({c["cls"] for c in cases})
    t4 = ["class,n,ASR_baseline,ASR_base_lo,ASR_base_hi,ASR_hardened,ASR_hard_lo,ASR_hard_hi,"
          "OTwrite_baseline,OTwrite_hardened"]
    def flat(d, cfg):
        return [v for cls in classes for v in d[cfg].get(cls, [])]
    for cls in classes + ["ALL"]:
        if cls == "ALL":
            sb, sh = flat(results, "baseline"), flat(results, "hardened")
            ob, oh = flat(otwrite, "baseline"), flat(otwrite, "hardened")
        else:
            sb, sh = results["baseline"].get(cls, []), results["hardened"].get(cls, [])
            ob, oh = otwrite["baseline"].get(cls, []), otwrite["hardened"].get(cls, [])
        n = len(sb)
        ab = wilson(sum(sb), n); ah = wilson(sum(sh), n)
        t4.append(f"{cls},{n},{ab[0]:.1f},{ab[1]:.1f},{ab[2]:.1f},{ah[0]:.1f},{ah[1]:.1f},{ah[2]:.1f},"
                  f"{100*sum(ob)/max(1,len(ob)):.1f},{100*sum(oh)/max(1,len(oh)):.1f}")
    with open(os.path.join(args.outdir, "table4.csv"), "w") as f:
        f.write("\n".join(t4) + "\n")

    # McNemar baseline vs hardened on pooled attack cases.
    base_all = flat(results, "baseline"); hard_all = flat(results, "hardened")
    b, c, pval = mcnemar(list(zip(base_all, hard_all)))

    # ---- Table 5: leave-one-out ablation ----
    t5 = ["config,overall_ASR,overall_ASR_lo,overall_ASR_hi,OTwrite_rate,benign_TSR,benign_TSR_lo,benign_TSR_hi"]
    for cfg in [c for c in ["baseline", "hardened"] + [f"-{m}" for m in ALL_MITIGATIONS] if c in configs]:
        s = flat(results, cfg); o = flat(otwrite, cfg); bres = benign_res[cfg]
        a = wilson(sum(s), len(s)); tsr = wilson(sum(bres), len(bres))
        t5.append(f"{cfg},{a[0]:.1f},{a[1]:.1f},{a[2]:.1f},{100*sum(o)/max(1,len(o)):.1f},"
                  f"{tsr[0]:.1f},{tsr[1]:.1f},{tsr[2]:.1f}")
    with open(os.path.join(args.outdir, "table5.csv"), "w") as f:
        f.write("\n".join(t5) + "\n")

    summary = {
        "overall_ASR_baseline_pct": round(100 * sum(base_all) / len(base_all), 1),
        "overall_ASR_hardened_pct": round(100 * sum(hard_all) / len(hard_all), 1),
        "OTwrite_baseline_pct": round(100 * sum(flat(otwrite, "baseline")) / len(base_all), 1),
        "OTwrite_hardened_pct": round(100 * sum(flat(otwrite, "hardened")) / len(hard_all), 1),
        "benign_TSR_baseline_pct": round(100 * sum(benign_res["baseline"]) / len(benign_res["baseline"]), 1),
        "benign_TSR_hardened_pct": round(100 * sum(benign_res["hardened"]) / len(benign_res["hardened"]), 1),
        "mcnemar_baseline_vs_hardened": {"fixed_by_hardening": b, "broken_by_hardening": c, "p_value": pval},
        "n_attack_cases": len(cases), "n_benign_tasks": len(benign), "reps": args.reps,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "provider": args.provider, "model": args.model, "base_url": args.base_url,
        "temperature": args.temperature, "reps": args.reps, "n_per_class": args.n_per_class,
        "retriever": args.retriever, "configs": list(configs), "carrier_source": carrier_source, "n_carriers": len(carriers),
        "mitigations": ALL_MITIGATIONS, "seed": args.seed,
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDataset source for injection carriers: {carrier_source} ({len(carriers)} carriers)")
    print(f"Wrote: {args.outdir}/table4.csv, table5.csv, summary.json, raw.jsonl, run_meta.json")


if __name__ == "__main__":
    main()
