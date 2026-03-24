#!/usr/bin/env bash
# Zip all price CSVs in data/prices/ into data/prices_<date>.zip
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRICES_DIR="$SCRIPT_DIR/prices"
OUT="$SCRIPT_DIR/prices_$(date +%Y%m%d).zip"

if [[ ! -d "$PRICES_DIR" ]]; then
    echo "ERROR: prices directory not found: $PRICES_DIR" >&2
    exit 1
fi

count=$(find "$PRICES_DIR" -name "*.csv" | wc -l | tr -d ' ')
if [[ "$count" -eq 0 ]]; then
    echo "No CSV files found in $PRICES_DIR" >&2
    exit 1
fi

echo "Zipping $count CSV file(s) → $OUT"
zip -j "$OUT" "$PRICES_DIR"/*.csv
echo "Done: $(du -h "$OUT" | cut -f1)"
