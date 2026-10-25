# ICS-LLM-TM: experiment harness

Threat-modelling testbed for MCP- and RAG-integrated LLM applications in an
industrial control environment. It simulates a textile dye-bath tank (T-101)
behind an OPC UA-style gateway, a RAG corpus of plant documents, and MCP tool
servers, then runs an attack suite (T1–T8) against a chosen LLM with and
without seven mitigations (M1–M7). It writes the paper's result tables.

This produces the numbers that replace every **[TBD]** in Tables 4 and 5 of the
manuscript.

## 1. Install

```bash
cd ics-llm-tm
python3 -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

## 2. Smoke-test (no network, ~1 min)

Validates the whole pipeline with a deterministic insecure mock agent:

```bash
python run_experiment.py --provider mock --reps 20
python make_figures.py
```

Expect baseline ASR ~100%, hardened much lower, benign TSR ~100%. These mock
numbers are **not** results — they only prove the harness works.

## 3. Real run (this is what goes in the paper)

Set an API key, then pick a model. Any OpenAI-compatible endpoint works.

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python run_experiment.py --provider openai --model gpt-4o-mini --reps 10

# Groq (Llama 3.1, OpenAI-compatible)
export OPENAI_API_KEY=gsk_...
python run_experiment.py --provider openai --model llama-3.1-8b-instant \
       --base-url https://api.groq.com/openai/v1 --reps 10

# Local Ollama or vLLM (no key cost)
export OPENAI_API_KEY=ollama
python run_experiment.py --provider openai --model llama3.1:8b \
       --base-url http://localhost:11434/v1 --reps 10

# Azure OpenAI: set --base-url to <your-endpoint>/openai/deployments/<deployment>/ ... /v1
```

Run **three** backbone models (e.g. one local open-weight + two API models) so
the paper can report per-model results. Use the same `--reps` for all.
Recommended: `--reps 20` for tight confidence intervals.

### Injection carriers (public data)

By default the harness pulls real injection phrasings from the Hugging Face
dataset **deepset/prompt-injections** (Apache-2.0) and attaches industrial
attacker goals to them. If that download is blocked, it falls back to a small
built-in list (recorded in `results/run_meta.json` as `carrier_source`). You
can also supply your own public CSV:

```bash
python run_experiment.py --provider openai --model gpt-4o-mini \
       --carriers-csv my_injections.csv        # columns: text[,label]
```

Always check `results/run_meta.json` → `carrier_source` says
`deepset/prompt-injections` (or your CSV), **not** `fallback`, before using the
numbers in the paper.

## 4. Outputs (in `results/`)

| File | Contents |
| --- | --- |
| `table4.csv` | ASR + OT-write by attack class, baseline vs hardened, with Wilson 95% CIs → paper Table 4 |
| `table5.csv` | Leave-one-out mitigation ablation (overall ASR, OT-write, benign TSR) → paper Table 5 |
| `summary.json` | Headline numbers + McNemar paired test (baseline vs hardened) |
| `raw.jsonl` | One line per run (config, rep, case id, success, ot_write) — full reproducibility |
| `run_meta.json` | Provider, model, reps, retriever, dataset source, seed |
| `carriers_used.txt` | The exact injection carriers used |
| `fig_asr_by_class.png`, `fig_ablation.png` | Figures |

## 5. What to send back

Zip the whole `results/` folder for **each** model you ran and share it. From
those I will:
- fill Tables 4 and 5 with real values and CIs,
- write the per-model supplementary tables,
- update the abstract/results prose and the figures,
- replace every [TBD] in the manuscript.

Tell me, for each run: the exact model name/version, provider, date, number of
reps, and the `carrier_source` line.

## 6. Mapping to the paper

| Code | Paper |
| --- | --- |
| `plant.py` | Zones Z6/Z7, safety envelope (Methods: Testbed) |
| `corpus.py` | Zone Z4 RAG pipeline |
| `mcp_tools.py` | Zone Z5 tools + Orchestrator (Z2) + mitigations M1–M7 |
| `attacks.py` | Attack suite T1–T8 + benign tasks |
| `payloads.py` | Public injection carriers |
| `models.py` | Backbone LLM wrappers |
| `run_experiment.py` | Configs, reps, metrics, Wilson + McNemar |

## 7. Notes / honesty

- The asset is **simulated**; results are a lower bound on real-plant risk
  (stated as a limitation in the manuscript).
- `--retriever minilm` uses MiniLM embeddings (needs `sentence-transformers`);
  default `tfidf` needs no extra download. Report which you used.
- Nothing here touches a real control system. Keep it that way.
