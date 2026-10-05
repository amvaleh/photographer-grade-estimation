#!/usr/bin/env bash
# Pull the 800px "medium" version of each listed photo from the production uploads dir.
# Input: TSV (photo_id<TAB>file_name), one per line, no header. Output: data/images/<photo_id>/<file>.
# Prefers medium_photo_<id>.webp, falls back to medium_<file_name> (older uploads predate the webp versions).
# Read-only on the server, niced, batched so it never holds the disk for long.
# Usage: SSH_HOST=<alias> UPLOADS_DIR=<path> scripts/fetch_images.sh data/pilot.tsv
set -euo pipefail
cd "$(dirname "$0")/.."
LIST=$1; BATCH=${BATCH:-1500}
: "${SSH_HOST:?set SSH_HOST (ssh alias of the file server)}"
: "${UPLOADS_DIR:?set UPLOADS_DIR (directory holding <photo_id>/medium_* on that server)}"
mkdir -p data/images data/tmp
rm -f data/tmp/batch_*; split -l "$BATCH" "$LIST" data/tmp/batch_
REMOTE='cd $UPLOADS_DIR
while IFS=$(printf "\t") read -r id name; do
  for f in "medium_photo_$id.webp" "medium_$name"; do
    if [ -f "$id/$f" ]; then printf "%s/%s\n" "$id" "$f"; break; fi
  done
done | tar -cf - -T -'
for b in data/tmp/batch_*; do
  for attempt in 1 2 3 4; do   # a dropped connection must not abort a multi-hour download
    if ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=6 "$SSH_HOST" "UPLOADS_DIR='$UPLOADS_DIR' nice -n 19 bash -c '$REMOTE'" < "$b" | tar -xf - -C data/images; then break; fi
    echo "batch $b failed (attempt $attempt), retrying in 30s"; sleep 30
  done
  echo "$b done: $(find data/images -type f | wc -l | tr -d ' ') files so far"
done
rm -rf data/tmp
