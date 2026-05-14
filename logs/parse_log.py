import re
import csv
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/d42481e.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results.csv"

pattern = re.compile(
    r"verdict=(\S+)\s+truth=(\S+)\s+correct=(\S+)\s+intent=(\S+)\s+predicted=(\S+)"
)

rows = []
with open(LOG) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            rows.append({
                "verdict":   m.group(1),
                "truth":     m.group(2),
                "correct":   m.group(3),
                "intent":    m.group(4),
                "predicted": m.group(5),
            })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["verdict", "truth", "correct", "intent", "predicted"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
