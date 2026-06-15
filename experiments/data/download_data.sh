#!/usr/bin/env bash
# Fetch the five UCI datasets into ./real/ (where every experiment expects them).
# UCI URLs occasionally change; if one 404s, find the dataset on archive.ics.uci.edu.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p real
cd real

BASE="https://archive.ics.uci.edu/ml/machine-learning-databases"
get() { [ -f "$2" ] && { echo "  have $2"; return; }; echo "  fetching $2"; curl -fL --retry 3 -o "$2" "$1"; }

echo "=== Census (Adult) ==="
get "$BASE/adult/adult.data" adult.data

echo "=== Wine Quality ==="
get "$BASE/wine-quality/winequality-red.csv"   winequality-red.csv
get "$BASE/wine-quality/winequality-white.csv" winequality-white.csv

echo "=== Forest (Covertype) ==="
if [ ! -f covtype.data ]; then
  curl -fL --retry 3 -o covtype.data.gz "$BASE/covtype/covtype.data.gz"
  gunzip -f covtype.data.gz
fi
echo "  have covtype.data"

echo "=== Power (household electric) ==="
if [ ! -f household_power_consumption.txt ]; then
  curl -fL --retry 3 -o power.zip "$BASE/00235/household_power_consumption.zip"
  unzip -o power.zip && rm -f power.zip
fi
echo "  have household_power_consumption.txt"

echo "=== Bike Sharing ==="
if [ ! -f hour.csv ]; then
  curl -fL --retry 3 -o bike.zip "$BASE/00275/Bike-Sharing-Dataset.zip"
  unzip -o bike.zip hour.csv day.csv && rm -f bike.zip
fi
echo "  have hour.csv"

echo "=== done. files in $(pwd): ==="
ls -la
