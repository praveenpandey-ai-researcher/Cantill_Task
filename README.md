# spendpipe

Config-driven spend data pipeline and classifier (Cantill.ai Data & ML task).

> Work in progress. Full setup and usage notes come at the end of the build.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # small CPU build
pip install -r requirements.txt
```

The first run downloads the embedding model (`BAAI/bge-m3`, ~2.2 GB) into `.cache/`.
For a smaller, faster model set `classify.model: intfloat/multilingual-e5-small` in the config.
Without internet, the classifier falls back to a local TF-IDF model automatically.

## Run

```bash
python -m spendpipe run --config configs/company_A.yaml --out outputs/A.csv
```

Data files are **not** in this repo. By default each config looks for data in
`../../company_<X>` relative to the config file; pass `--data-dir <folder>` to
point at another folder.
