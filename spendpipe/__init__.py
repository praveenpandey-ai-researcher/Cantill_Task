"""spendpipe: config-driven spend data pipeline and classifier.

Stages (each a module):
    config      -> load and validate a company YAML config
    ingest      -> read source files, map columns to a common schema
    clean       -> amounts (VAT, FX), dates
    match       -> cross-system matching / count-once (DUPLICATE)
    exclude     -> status and NON_PROCUREMENT rules
    dedup       -> supplier_key and item_key
    classify    -> zero-shot tree classification with confidence
    output      -> final CSV + data-quality report
"""

__version__ = "0.1.0"
