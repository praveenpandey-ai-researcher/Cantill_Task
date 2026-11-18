# DECISIONS

Data forensics and every assumption made. Written as the work happens.

## 0. Setup
- Repo root is `Cantill_Task/`. Task data stays outside the repo (`../company_A`, `../company_B`)
  and `.gitignore` blocks all `*.csv` except `outputs/` and `submission/`.
- Config file paths are relative to `data_dir`; `--data-dir` overrides it so the
  blind-test files (A2, B2) run with the unchanged A and B configs.
- Dependency versions are pinned. On the build PC, Windows Smart App Control blocked
  the compiled files of the newest pandas, scikit-learn and tokenizers wheels
  ("An Application Control policy has blocked this file"). Slightly older, widely used
  releases load fine, so `requirements.txt` pins those (pandas 2.2.3, scikit-learn
  1.5.2, torch 2.6.0 CPU, tokenizers 0.21.0, sentence-transformers 3.3.1).
