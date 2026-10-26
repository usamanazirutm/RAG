@echo off
rem Example daily runner for the Groq free tier.
rem Copy to run_day.bat (which is git-ignored) and set your key once with:
rem     setx OPENAI_API_KEY gsk_your_key_here
rem Never commit a file that contains your API key.
cd /d "%~dp0"
call .venv\Scripts\activate
set G=--provider openai --base-url https://api.groq.com/openai/v1 --max-tokens 2000
python run_experiment.py %G% --model openai/gpt-oss-20b --configs baseline,hardened --reps 3 --outdir results/results_gptoss20b
python run_experiment.py %G% --model openai/gpt-oss-20b --reps 1 --outdir results/results_gptoss20b_ablation
python run_experiment.py %G% --model openai/gpt-oss-120b --configs baseline,hardened --reps 3 --outdir results/results_gptoss120b
python run_experiment.py %G% --model qwen/qwen3.8-27b --configs baseline,hardened --reps 3 --outdir results/results_qwen
pause
