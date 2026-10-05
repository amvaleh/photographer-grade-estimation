#!/usr/bin/env bash
# Read-only export of the two manifests from production into data/ (git-ignored).
# Usage: SSH_HOST=<alias> DB_NAME=<db> scripts/export_manifest.sh [photos_per_photographer=24]
set -euo pipefail
cd "$(dirname "$0")/.."
PER=${1:-24}
: "${SSH_HOST:?set SSH_HOST to the ssh alias of the host running Postgres}"
: "${DB_NAME:?set DB_NAME to the Postgres database name}"
mkdir -p data
run() {  # $1 = sql file, $2 = output csv
  { echo "SET default_transaction_read_only = on; SET statement_timeout = '180s';"
    echo "COPY ("; cat "$1"; echo ") TO STDOUT WITH CSV HEADER;"; } \
    | ssh "$SSH_HOST" "nice -n 19 psql -q -d $DB_NAME -v ON_ERROR_STOP=1 -v per_photographer=$PER" 2>/dev/null > "$2"
  echo "$2: $(($(wc -l < "$2") - 1)) rows"
}
run sql/photographers.sql data/photographers.csv
run sql/photos.sql        data/photos.csv
run sql/profile.sql       data/profile.csv
run sql/gear.sql          data/gear.csv
