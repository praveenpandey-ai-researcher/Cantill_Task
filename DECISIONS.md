# DECISIONS

Data forensics and every assumption made. Written as the work happens.

## 0. Setup
- Repo root is `Cantill_Task/`. Task data stays outside the repo (`../company_A`, `../company_B`)
  and `.gitignore` blocks all `*.csv` except `outputs/` and `submission/`.
- Config file paths are relative to `data_dir`; `--data-dir` overrides it so the
  blind-test files (A2, B2) run with the unchanged A and B configs.
