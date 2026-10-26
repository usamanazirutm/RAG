# Results

Place the four run folders here exactly as produced by `run_experiment.py`:

| Folder | Model | Configurations | Reps | Attack runs |
| --- | --- | --- | --- | --- |
| `results_gptoss20b` | openai/gpt-oss-20b | baseline, hardened | 3 | 264 |
| `results_gptoss120b` | openai/gpt-oss-120b | baseline, hardened | 3 | 264 |
| `results_qwen` | qwen/qwen3.8-27b | baseline, hardened | 3 | 264 |
| `results_gptoss20b_ablation` | openai/gpt-oss-20b | all 9 (leave-one-out) | 1 | 396 |

Each folder contains:

- `raw.jsonl` — one line per run (config, rep, case id, class, success, ot_write); the primary data
- `table4.csv` — attack success and OT-write by class, baseline vs hardened, with Wilson 95% CIs
- `table5.csv` — overall attack success, OT-write and benign task success per configuration
- `summary.json` — headline numbers and McNemar test
- `run_meta.json` — model, provider, settings and dataset source
- `carriers_used.txt` — the exact injection phrasings used

All tables can be regenerated from `raw.jsonl` by rerunning the same command
with the same `--outdir` (completed runs are skipped and statistics rebuilt).
