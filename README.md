# spendpipe

Config-driven spend data pipeline and classifier (Cantill.ai Data & ML task).

> Work in progress. Full setup and usage notes come at the end of the build.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run

```bash
python -m spendpipe run --config configs/company_A.yaml --out outputs/A.csv
```

Data files are **not** in this repo. By default each config looks for data in
`../../company_<X>` relative to the config file; pass `--data-dir <folder>` to
point at another folder.
