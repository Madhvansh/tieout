#!/usr/bin/env bash
# Retry the filings whose hosts refused the python client, using curl with a full browser header set.
cd "$(dirname "$0")/.."
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
grep "FAILED" data/manifest.csv | while IFS=, read -r doc ticker rest; do
  url=$(python - "$doc" <<'PY'
import csv, sys
for r in csv.DictReader(open("data/manifest.csv", newline="", encoding="utf-8")):
    if r["doc_id"] == sys.argv[1]:
        print(r["pdf_url"])
PY
)
  out="data/pdf/$doc.pdf"
  code=$(curl -sS -L --compressed --max-time 120 -o "$out.part" -w "%{http_code}" \
    -H "User-Agent: $UA" \
    -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8" \
    -H "Accept-Language: en-US,en;q=0.9" \
    -H "Sec-Fetch-Dest: document" -H "Sec-Fetch-Mode: navigate" -H "Sec-Fetch-Site: none" \
    -H "Upgrade-Insecure-Requests: 1" "$url")
  head=$(head -c 5 "$out.part" 2>/dev/null)
  if [ "$code" = "200" ] && [ "$head" = "%PDF-" ]; then
    mv "$out.part" "$out"; echo "ok   $doc $(du -h "$out" | cut -f1)"
  else
    rm -f "$out.part"; echo "FAIL $doc http=$code"
  fi
done
