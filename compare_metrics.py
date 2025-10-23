# compare_metrics.py
import json
from pathlib import Path

p1 = Path("metrics_phase1.json")
p2 = Path("metrics_phase2.json")
if not p1.exists() or not p2.exists():
    print("One or both metrics files missing. Make sure phase1 and phase2 have run.")
    raise SystemExit(1)

m1 = json.load(open(p1))
m2 = json.load(open(p2))

common = sorted(set(m1.keys()).intersection(set(m2.keys())), key=lambda x: int(x))
mismatches = []
for e in common:
    if m1[e] != m2[e]:
        mismatches.append((e, m1[e], m2[e]))

if not mismatches:
    print("✅ Metrics identical for common epochs between phase1 and phase2.")
else:
    print("❌ MISMATCHES FOUND:")
    for e,a,b in mismatches:
        print(f"Epoch {e}\n Phase1: {a}\n Phase2: {b}\n")
