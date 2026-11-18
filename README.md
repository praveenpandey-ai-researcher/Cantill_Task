# spendpipe

A config-driven spend data pipeline and classifier. Give it a company's raw purchase data and
its category tree, and it produces one clean spend cube:
- every line counted once
- tax and currency handled
- non-spend excluded with a reason code
- suppliers and items de-duplicated
- every included line classified into the company's tree, with a confidence

A new company needs a **new config file, not new code**. No category names are in the code.

- How it works and the measured accuracy: [APPROACH.md](APPROACH.md)
- Every data finding and assumption, with numbers: [DECISIONS.md](DECISIONS.md)
- Step-by-step for the 2-hour test: [BLINDTEST.md](BLINDTEST.md)

## Setup

Python 3.10 to 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # small CPU build
pip install -r requirements.txt
python -m pytest -q             # 131 tests, no data or model download needed
```

The first run downloads the embedding model `BAAI/bge-m3` (~2.2 GB) into `.cache/`. After
that it works offline, and every embedded text is cached on disk. If the model can't be loaded
(no internet, blocked files), the classifier switches to a local TF-IDF model automatically
and the run still completes. For a smaller, faster model, set
`classify.model: intfloat/multilingual-e5-small` in a config.

**No API key is needed.** The tool is fully local and makes no LLM calls.

## The one command

```bash
python -m spendpipe run --config configs/company_A.yaml --out outputs/praveenpandey__A.csv
python -m spendpipe run --config configs/company_B.yaml --out outputs/praveenpandey__B.csv
```

Each run writes:
- the output CSV: `record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2,
  l3, l4, confidence`, one row per input record
- a data-quality report next to it (`<name>_report.json`): exclusions by reason, spend by L1,
  unparsed or implausible dates, FX problems, match rates, timings

Data for new records with the same layout (A2, B2) uses the same config:

```bash
python -m spendpipe run --config configs/company_A.yaml --data-dir <folder with A2 files> --out submission/praveenpandey__A2.csv
```

A new company: draft its config from the file, review it, then run:

```bash
python -m spendpipe profile <C invoice file> --tree <C tree file> --out configs/company_C.yaml
python -m spendpipe run --config configs/company_C.yaml --out submission/praveenpandey__C.csv
```

## Data is not in this repo

The task data is confidential. `.gitignore` blocks every `*.csv` except the pipeline outputs.
By default the configs read data from `../../company_A` and `../../company_B`, relative to the
config file, which is the kit's folder layout. Point elsewhere with `--data-dir`.

## Layout

```
spendpipe/        the package
  cli.py            run / profile commands
  config.py         YAML loading, defaults, validation
  ingest.py         read files, map columns to canonical fields
  clean.py          amounts, VAT, FX, dates, text
  exclude.py        status rules, NON_PROCUREMENT rules, precedence
  match.py          count once: duplicates and cross-system matching
  normalise.py      PO / invoice / supplier / text keys
  dedup.py          supplier_key and item_key
  taxonomy.py       category tree from file
  embed.py          embeddings with disk cache and TF-IDF fallback
  classify.py       zero-shot classifier, priors, kNN memory, calibration
  output.py         output contract and data-quality report
  profile.py        draft a config for a new file
configs/          one YAML per company + shared lexicons (non-procurement, abbreviations)
analysis/         evaluation scripts (print aggregates only)
tests/            pytest unit tests (synthetic data only)
outputs/          A and B outputs + reports
```

## Evaluation scripts

```bash
python analysis/explore_data.py                                  # data forensics behind DECISIONS.md
python analysis/eval_matching.py                                 # count-once evidence for A
python analysis/eval_exclusions.py --config configs/company_B.yaml   # exclusions vs B seed labels
python analysis/eval_dedup.py                                    # supplier/item pairwise P/R vs A keys
python analysis/eval_classify.py --company A B                   # classification, labels hidden
python analysis/eval_classify.py --company A B --reference-split 0.5   # with half the labels as memory
python analysis/eval_classify.py --fit-calibration               # refit confidence curves
```

## AI tools used

Built with **Claude Code** (Anthropic) as a pair-programmer: it helped with code, data
exploration queries and documentation. Every design choice was checked against measurements
on the data, which are recorded in DECISIONS.md. The tool itself uses no LLM at run time.
